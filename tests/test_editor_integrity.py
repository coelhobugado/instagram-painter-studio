"""Regressões de identidade de tinta, edição espacial e preparação do pincel."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import cv2
import numpy as np

from studio.adb_manager import ADBManager
from studio.device.ui_timing import UITimingManager
from studio.execution_worker import ExecutionWorker
from studio.device.quality_gate import QualityGateResult
from studio.painter_engine import PainterEngine
from studio.app_gui import InstagramPainterStudioGUI
from studio.storage.editable_project import load_project, save_project
from studio.vision.perceptual_color import PerceptualColorEngine


class EditorIntegrityTests(unittest.TestCase):
    def test_line_strokes_can_be_selected_and_recolored_independently(self):
        engine = PainterEngine()
        fit = np.full((40, 50, 3), 22, dtype=np.uint8)
        white = engine.color_engine.match_color((255, 255, 255))
        strokes = [[(5, 8), (20, 8)], [(30, 28), (45, 28)]]
        plan = {
            "mode": "line_art", "fit_image": fit, "target_executable": fit.copy(),
            "palette_colors": [{"name": "Branco", "rgb": (255, 255, 255),
                                "hex": "#FFFFFF", "matched_hex": "#FFFFFF"}],
            "layers": [{"name": "Linhas", "purpose": "line_art_centerline", "brush": "solid",
                        "thickness": "min", "color_rgb": (255, 255, 255),
                        "render_color_rgb": tuple(white["matched_rgb"]), "touch_info": white,
                        "strokes": strokes}],
            "seed": 42, "time_budget_s": 600.0,
        }
        engine._attach_stroke_editable_regions(plan)
        first_id = int(plan["region_map"][8, 10])
        second_id = int(plan["region_map"][28, 35])
        self.assertGreater(first_id, 0)
        self.assertGreater(second_id, 0)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(int(plan["region_map"][18, 25]), 0)
        self.assertEqual(plan["layers"][0]["stroke_region_ids"], [first_id, second_id])

        revised = engine.recolor_region(plan, second_id, (255, 0, 0))
        self.assertEqual(revised["layers"][0]["color_rgb"], (255, 255, 255))
        self.assertEqual(revised["layers"][1]["color_rgb"], (255, 0, 0))
        self.assertEqual(revised["editable_regions"][1]["override_rgb"], (255, 0, 0))

    def test_line_background_and_ink_have_separate_click_regions(self):
        engine = PainterEngine()
        fit = np.full((30, 40, 3), 22, dtype=np.uint8)
        plan = {
            "mode": "line_art", "fit_image": fit,
            "palette_colors": [{"hex": "#000000"}, {"hex": "#FFFFFF"}],
            "layers": [
                {"purpose": "background", "strokes": [[(0, 5), (39, 5)]]},
                {"purpose": "line_art_centerline", "thickness": "min",
                 "strokes": [[(5, 15), (30, 15)]]},
            ],
        }
        engine._attach_stroke_editable_regions(plan)
        background_id = int(plan["region_map"][25, 20])
        ink_id = int(plan["region_map"][15, 20])
        self.assertNotEqual(background_id, ink_id)
        self.assertEqual(plan["layers"][0]["palette_index"], 0)
        self.assertEqual(plan["layers"][1]["palette_index"], 1)
        self.assertEqual(plan["layers"][0]["stroke_region_ids"], [background_id])
        self.assertEqual(plan["layers"][1]["stroke_region_ids"], [ink_id])

    def test_preview_click_selects_line_stroke_from_scaled_image(self):
        region_map = np.zeros((20, 40), dtype=np.int32)
        region_map[8:12, 25:30] = 7
        view = SimpleNamespace(
            current_plan={"region_map": region_map, "editable_regions": [{"id": 7}]},
            _editing_plan=False,
            _preview_display_size=(200, 100),
            lbl_canvas_preview=SimpleNamespace(winfo_width=lambda: 240, winfo_height=lambda: 140,
                                               winfo_rootx=lambda: 100, winfo_rooty=lambda: 200),
            _selected_region_id=None, _selected_region_ids=set(),
            _refresh_edit_controls=Mock(),
            _render_selected_preview=Mock(),
        )
        view._preview_event_to_image = lambda event: InstagramPainterStudioGUI._preview_event_to_image(view, event)
        view._select_preview_bounds = lambda start, end, additive=False: InstagramPainterStudioGUI._select_preview_bounds(
            view, start, end, additive
        )
        # O evento vem do label interno, deslocado dentro do componente externo.
        InstagramPainterStudioGUI._on_preview_click(
            view, SimpleNamespace(x=135, y=48, x_root=255, y_root=268, state=0)
        )
        self.assertEqual(view._selected_region_id, 7)
        self.assertEqual(view._selected_region_ids, {7})
        view._refresh_edit_controls.assert_called_once()
        view._render_selected_preview.assert_called_once()

    def test_drag_region_selection_collects_many_parts_and_ctrl_adds(self):
        region_map = np.zeros((20, 30), dtype=np.int32)
        region_map[3:6, 3:7] = 1
        region_map[8:11, 12:16] = 2
        region_map[15:18, 24:27] = 3
        view = SimpleNamespace(
            current_plan={"region_map": region_map,
                          "editable_regions": [{"id": 1}, {"id": 2}, {"id": 3}]},
            _selected_region_ids=set(), _selected_region_id=None,
            _refresh_edit_controls=Mock(), _render_selected_preview=Mock(),
        )
        InstagramPainterStudioGUI._select_preview_bounds(view, (1, 1), (18, 12))
        self.assertEqual(view._selected_region_ids, {1, 2})
        InstagramPainterStudioGUI._select_preview_bounds(view, (22, 14), (29, 19), additive=True)
        self.assertEqual(view._selected_region_ids, {1, 2, 3})
        self.assertEqual(view._refresh_edit_controls.call_count, 2)

    def test_drag_over_ink_does_not_recolor_background(self):
        region_map = np.ones((20, 30), dtype=np.int32)
        region_map[5:8, 8:14] = 2
        view = SimpleNamespace(
            current_plan={"region_map": region_map, "editable_regions": [
                {"id": 1, "selection_role": "background"}, {"id": 2},
            ]},
            _selected_region_ids=set(), _selected_region_id=None,
            _refresh_edit_controls=Mock(), _render_selected_preview=Mock(),
        )
        InstagramPainterStudioGUI._select_preview_bounds(view, (2, 2), (17, 12))
        self.assertEqual(view._selected_region_ids, {2})
        InstagramPainterStudioGUI._select_preview_bounds(view, (2, 2), (2, 2))
        self.assertEqual(view._selected_region_ids, set())
        self.assertIn("Selecionar fundo", view._selection_notice)
        view.current_plan["mode"] = "line_art"
        view.current_plan["editable_regions"][0]["name"] = "Fundo"
        InstagramPainterStudioGUI.action_select_background(view)
        self.assertEqual(view._selected_region_ids, {1})

    def test_preview_mouse_drag_selects_regions_on_release(self):
        region_map = np.zeros((20, 30), dtype=np.int32)
        region_map[3:6, 3:7] = 1
        region_map[8:11, 12:16] = 2
        view = SimpleNamespace(
            current_plan={"region_map": region_map,
                          "editable_regions": [{"id": 1}, {"id": 2}]},
            _editing_plan=False, _preview_display_size=(300, 200),
            lbl_canvas_preview=SimpleNamespace(
                winfo_width=lambda: 300, winfo_height=lambda: 200,
                winfo_rootx=lambda: 100, winfo_rooty=lambda: 200,
            ),
            _selection_start=None, _selection_drag=None,
            _selected_region_ids=set(), _selected_region_id=None,
            _refresh_edit_controls=Mock(), _render_selected_preview=Mock(),
        )
        view._preview_event_to_image = lambda event, clamp=False: InstagramPainterStudioGUI._preview_event_to_image(
            view, event, clamp
        )
        view._select_preview_bounds = lambda start, end, additive=False: InstagramPainterStudioGUI._select_preview_bounds(
            view, start, end, additive
        )
        InstagramPainterStudioGUI._on_preview_press(view, SimpleNamespace(x_root=120, y_root=220))
        InstagramPainterStudioGUI._on_preview_drag(view, SimpleNamespace(x_root=270, y_root=320))
        InstagramPainterStudioGUI._on_preview_release(view, SimpleNamespace(x_root=270, y_root=320, state=0))
        self.assertEqual(view._selected_region_ids, {1, 2})
        self.assertIsNone(view._selection_start)
        self.assertIsNone(view._selection_drag)

    def test_color_button_dispatches_whole_selection_once(self):
        engine = Mock()
        view = SimpleNamespace(
            _selected_region_ids={2, 1},
            current_plan={"editable_regions": [
                {"id": 1, "palette_index": 0, "name": "Traço 1"},
                {"id": 2, "palette_index": 0, "name": "Traço 2"},
            ], "palette_colors": [{"hex": "#FFFFFF"}]},
            engine=engine, _run_plan_edit=Mock(),
        )
        with patch("studio.app_gui.colorchooser.askcolor", return_value=((10, 20, 30), "#0A141E")):
            InstagramPainterStudioGUI.action_change_region_color(view)
        view._run_plan_edit.assert_called_once()
        edit = view._run_plan_edit.call_args.args[0]
        edit(view.current_plan)
        engine.recolor_regions.assert_called_once_with(view.current_plan, (1, 2), (10, 20, 30))

    def test_bulk_recolor_changes_selected_strokes_in_one_revision(self):
        engine = PainterEngine()
        fit = np.full((45, 55, 3), 22, dtype=np.uint8)
        white = engine.color_engine.match_color((255, 255, 255))
        strokes = [[(4, 8), (16, 8)], [(21, 20), (34, 20)], [(39, 34), (51, 34)]]
        plan = {
            "mode": "line_art", "fit_image": fit, "target_executable": fit.copy(),
            "palette_colors": [{"name": "Branco", "rgb": (255, 255, 255),
                                "hex": "#FFFFFF", "matched_hex": "#FFFFFF"}],
            "layers": [{"name": "Linhas", "purpose": "line_art_centerline", "brush": "solid",
                        "thickness": "min", "color_rgb": (255, 255, 255),
                        "render_color_rgb": tuple(white["matched_rgb"]), "touch_info": white,
                        "strokes": strokes}],
            "seed": 42, "time_budget_s": 600.0,
        }
        engine._attach_stroke_editable_regions(plan)
        ids = plan["layers"][0]["stroke_region_ids"]
        revised = engine.recolor_regions(plan, ids[:2], (255, 0, 0))
        self.assertEqual(revised["revision"], 1)
        self.assertEqual(revised["layers"][0]["color_rgb"], (255, 0, 0))
        self.assertEqual(revised["layers"][0]["stroke_region_ids"], ids[:2])
        self.assertEqual(revised["layers"][1]["color_rgb"], (255, 255, 255))
        self.assertEqual(revised["layers"][1]["stroke_region_ids"], ids[2:])
        self.assertEqual(plan["layers"][0]["color_rgb"], (255, 255, 255))
        self.assertEqual(sum("override_rgb" in r for r in revised["editable_regions"]), 2)

    def test_same_coordinates_on_other_page_are_distinct(self):
        first = {"palette_page": 1, "target_x": 411, "target_y": 2184,
                 "is_long_press": False, "id": 1}
        second = dict(first, palette_page=2, id=2)
        self.assertNotEqual(PerceptualColorEngine.physical_touch_key(first),
                            PerceptualColorEngine.physical_touch_key(second))

    def test_physical_palette_choice_keeps_requested_page_and_entry(self):
        engine = PerceptualColorEngine()
        entry = next(e for e in engine.entries if e.get("palette_page") == 3)
        selected = engine.match_color(tuple(entry["rgb"]), preferred_lut_id=entry["id"])
        self.assertEqual(selected["id"], entry["id"])
        self.assertEqual(selected["palette_page"], 3)
        self.assertEqual(tuple(selected["matched_rgb"]), tuple(entry["rgb"]))

    def test_constrained_quantization_uses_physical_lab(self):
        engine = PainterEngine()
        image = np.zeros((6, 6, 3), dtype=np.uint8)
        image[:, :3] = (255, 255, 255)
        image[:, 3:] = (38, 38, 38)
        target, _ideal, labels, palette = engine.quantize_instagram_constrained(image, num_colors=2)
        self.assertEqual(target.shape, image.shape)
        self.assertEqual(labels.shape, image.shape[:2])
        self.assertEqual(len(palette), 2)
        self.assertNotEqual(int(labels[0, 0]), int(labels[0, 5]))
        self.assertGreater(int(target[0, 0, 0]), int(target[0, 5, 0]))

    def test_region_edit_is_isolated_and_project_roundtrips(self):
        engine = PainterEngine()
        # Plano pequeno com dois componentes da mesma cor e traços independentes.
        fit = np.full((20, 30, 3), 22, dtype=np.uint8)
        region_map = np.zeros((20, 30), dtype=np.int32)
        region_map[3:12, 2:11] = 1
        region_map[3:12, 18:27] = 2
        layers = [{
            "name": "Camada 1", "color_rgb": (255, 255, 255),
            "render_color_rgb": (255, 255, 255), "touch_info": engine.color_engine.match_color((255, 255, 255)),
            "brush": "solid", "purpose": "color_mass", "thickness": "min", "palette_index": 0,
            "strokes": [[(3, 5), (10, 5)], [(19, 5), (26, 5)]],
            "stroke_region_ids": [1, 2],
        }]
        plan = {"mode": "layered_paint", "fit_image": fit, "target_executable": fit.copy(),
                "region_map": region_map, "editable_regions": [
                    {"id": 1, "name": "Área 1", "palette_index": 0, "area": 81, "bbox": (2, 3, 9, 9)},
                    {"id": 2, "name": "Área 2", "palette_index": 0, "area": 81, "bbox": (18, 3, 9, 9)},
                ], "palette_colors": [{"name": "Branco", "rgb": (255, 255, 255),
                                      "hex": "#FFFFFF", "matched_hex": "#FFFFFF"}],
                "layers": layers, "seed": 42, "time_budget_s": 600.0,
                "custom_bounds": {"left": 100, "top": 500, "right": 130, "bottom": 520,
                                  "center_x": 115, "center_y": 510, "max_width": 30, "max_height": 20}}
        revised = engine.recolor_region(plan, 1, (255, 0, 0))
        self.assertEqual(plan["layers"][0]["color_rgb"], (255, 255, 255))
        self.assertEqual(revised["layers"][0]["color_rgb"], (255, 0, 0))
        self.assertEqual(revised["layers"][1]["color_rgb"], (255, 255, 255))
        np.testing.assert_array_equal(revised["target_executable"][region_map == 2],
                                      plan["target_executable"][region_map == 2])
        self.assertEqual(revised["compiled_motion_plan"].total_strokes, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.npz"
            save_project(path, revised)
            restored = load_project(path, engine)
        self.assertEqual(restored["plan_hash"], revised["plan_hash"])
        self.assertEqual(restored["custom_bounds"], plan["custom_bounds"])
        np.testing.assert_array_equal(restored["target_executable"], revised["target_executable"])

    def test_two_disconnected_objects_of_same_color_stay_independent(self):
        image = np.full((80, 80, 3), (220, 60, 30), dtype=np.uint8)
        image[10:30, 10:30] = (20, 20, 230)
        image[45:65, 45:65] = (20, 20, 230)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "two_objects.png"
            cv2.imwrite(str(source), image)
            engine = PainterEngine()
            plan = engine.generate_layered_paint_plan(
                str(source), num_colors=2, brush_spacing=3, min_area=10,
                include_contours=False, enable_semantic_highlights=False,
                enable_colored_details=False, optimize_in_simulation=False,
                custom_bounds={"max_width": 160, "max_height": 160,
                               "center_x": 540, "center_y": 1180},
            )
        first_id = int(plan["region_map"][40, 40])
        second_id = int(plan["region_map"][110, 110])
        self.assertGreater(first_id, 0)
        self.assertGreater(second_id, 0)
        self.assertNotEqual(first_id, second_id)
        revised = engine.recolor_region(plan, first_id, (40, 220, 80))
        second_mask = plan["region_map"] == second_id
        np.testing.assert_array_equal(revised["target_executable"][second_mask],
                                      plan["target_executable"][second_mask])
        self.assertFalse(np.array_equal(revised["target_executable"][40, 40],
                                        plan["target_executable"][40, 40]))
        full_render, _ = engine.simulator.render_simulation(
            revised["fit_image"].shape, revised["layers"], background_bgr=(22, 22, 22)
        )
        np.testing.assert_array_equal(revised["simulation_image"], full_render)

    def test_slider_confirmation_requires_both_positions_to_change(self):
        before = np.zeros((200, 100, 3), dtype=np.uint8)
        after = before.copy()
        cv2.circle(before, (40, 40), 10, (255, 255, 255), -1)
        cv2.circle(after, (40, 160), 10, (255, 255, 255), -1)
        self.assertTrue(ADBManager._slider_changed_at_targets(before, after, 40, 40, 160))
        self.assertFalse(ADBManager._slider_changed_at_targets(before, before, 40, 40, 160))

    def test_real_manager_prepares_minimum_before_drawing(self):
        manager = ADBManager(adb_path="unused", device_target="OFFLINE")
        manager.run_cmd = Mock(return_value=(True, "", ""))
        manager.profile_loader.get_slider_thickness = Mock(return_value={
            "x": 40, "min_y": 160, "medium_y": 100, "thick_y": 40,
        })
        first = np.zeros((200, 100, 3), dtype=np.uint8)
        second = first.copy()
        cv2.circle(first, (40, 40), 10, (255, 255, 255), -1)
        cv2.circle(second, (40, 160), 10, (255, 255, 255), -1)
        manager.capture_screenshot = Mock(side_effect=[first, second])
        manager.timing.settle = Mock()
        self.assertTrue(manager.prepare_brush_thickness("min"))
        commands = [call.args[0][1] for call in manager.run_cmd.call_args_list]
        self.assertEqual(commands, ["input tap 40 40", "input tap 40 160"])

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    def test_worker_never_draws_after_thickness_confirmation_failure(self, _export):
        adb = Mock()
        adb.timing = UITimingManager(is_mock=True)
        adb.supports_visual_thickness_check = True
        adb.prepare_brush_thickness.side_effect = RuntimeError("slider não confirmado")
        adb.auto_reconnect.return_value = False
        adb.capture_screenshot.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        plan = {"total_strokes": 1, "quality_gate": QualityGateResult(passed=True, summary="Plano de teste aprovado"), "layers": [{
            "brush": "solid", "thickness": "min", "touch_info": {"target_x": 411, "target_y": 2184},
            "strokes": [[(1, 1), (20, 20)]],
        }]}
        outcomes = []
        worker = ExecutionWorker(adb, plan, {"on_finished": lambda *args: outcomes.append(args)})
        worker.state_machine.validate_safety_for_drawing = Mock(return_value=(True, "OK"))
        worker.start()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(adb.execute_stroke_batch.call_count, 0)
        self.assertFalse(outcomes[0][0])


if __name__ == "__main__":
    unittest.main()
