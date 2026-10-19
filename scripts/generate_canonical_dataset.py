import os
import shutil
import cv2
import numpy as np

def generate_canonical_dataset(output_dir="benchmarks/canonical"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. line_1px.png: 1px horizontal, vertical and diagonal lines (black on white)
    img_1px = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img_1px, (20, 50), (180, 50), (0, 0, 0), 1)
    cv2.line(img_1px, (100, 20), (100, 180), (0, 0, 0), 1)
    cv2.line(img_1px, (30, 170), (170, 30), (0, 0, 0), 1)
    cv2.imwrite(os.path.join(output_dir, "line_1px.png"), img_1px)
    
    # 2. line_2px.png: 2px parallel lines
    img_2px = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img_2px, (30, 60), (170, 60), (0, 0, 0), 2)
    cv2.line(img_2px, (30, 100), (170, 100), (0, 0, 0), 2)
    cv2.line(img_2px, (30, 140), (170, 140), (0, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "line_2px.png"), img_2px)
    
    # 3. line_curves.png: smooth sinusoidal and circular curves
    img_curves = np.full((300, 300, 3), 255, dtype=np.uint8)
    xs = np.linspace(20, 280, 200)
    ys = (150 + 60 * np.sin(xs * 0.05)).astype(np.int32)
    pts = np.column_stack((xs.astype(np.int32), ys)).reshape((-1, 1, 2))
    cv2.polylines(img_curves, [pts], False, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(img_curves, (150, 150), 70, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "line_curves.png"), img_curves)
    
    # 4. line_t_junction.png: crisp T-junction
    img_t = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img_t, (40, 100), (160, 100), (0, 0, 0), 2) # horizontal bar
    cv2.line(img_t, (100, 100), (100, 170), (0, 0, 0), 2) # vertical stem
    cv2.imwrite(os.path.join(output_dir, "line_t_junction.png"), img_t)
    
    # 5. line_x_junction.png: crisp X-junction crossing
    img_x = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img_x, (40, 40), (160, 160), (0, 0, 0), 2)
    cv2.line(img_x, (160, 40), (40, 160), (0, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "line_x_junction.png"), img_x)
    
    # 6. line_loop.png: closed loop
    img_loop = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.ellipse(img_loop, (100, 100), (60, 40), 30, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "line_loop.png"), img_loop)
    
    # 7. flower_lineart.png: botanical line art
    if os.path.exists("inputs/flores.jpg"):
        shutil.copy("inputs/flores.jpg", os.path.join(output_dir, "flower_lineart.png"))
    else:
        cv2.imwrite(os.path.join(output_dir, "flower_lineart.png"), img_curves)
        
    # 8. anime_face.png: stylized anime face contours
    img_anime = np.full((300, 300, 3), 255, dtype=np.uint8)
    # Face chin outline
    chin = np.array([[80, 120], [100, 210], [150, 250], [200, 210], [220, 120]], dtype=np.int32)
    cv2.polylines(img_anime, [chin], False, (0, 0, 0), 2, cv2.LINE_AA)
    # Eyes
    cv2.ellipse(img_anime, (115, 150), (22, 12), -10, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(img_anime, (185, 150), (22, 12), 10, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    # Pupils
    cv2.circle(img_anime, (115, 150), 6, (0, 0, 0), -1)
    cv2.circle(img_anime, (185, 150), 6, (0, 0, 0), -1)
    # Catchlights
    cv2.circle(img_anime, (113, 148), 2, (255, 255, 255), -1)
    cv2.circle(img_anime, (183, 148), 2, (255, 255, 255), -1)
    # Nose & Mouth
    cv2.line(img_anime, (150, 175), (148, 185), (0, 0, 0), 2)
    cv2.ellipse(img_anime, (150, 215), (20, 8), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "anime_face.png"), img_anime)
    
    # 9. manga_hatching.png: parallel and cross hatching
    img_hatch = np.full((250, 250, 3), 255, dtype=np.uint8)
    for i in range(20, 230, 8):
        cv2.line(img_hatch, (i, 20), (i + 40, 100), (0, 0, 0), 1) # simple hatch
    for i in range(20, 200, 10):
        cv2.line(img_hatch, (i, 130), (i + 40, 230), (0, 0, 0), 1)
        cv2.line(img_hatch, (i + 40, 130), (i, 230), (0, 0, 0), 1) # cross hatch
    cv2.imwrite(os.path.join(output_dir, "manga_hatching.png"), img_hatch)
    
    # 10. logo.png: crisp geometric corporate logo with multiple colors
    img_logo = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(img_logo, (150, 150), 90, (56, 151, 240), -1) # Blue
    cv2.rectangle(img_logo, (100, 100), (200, 200), (237, 73, 86), -1) # Red
    cv2.circle(img_logo, (150, 150), 40, (255, 200, 64), -1) # Yellow
    cv2.circle(img_logo, (150, 150), 20, (255, 255, 255), -1) # White center
    cv2.imwrite(os.path.join(output_dir, "logo.png"), img_logo)
    
    # 11. text.png: stylized typography lines
    img_text = np.full((200, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img_text, "INSTAGRAM", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_text, "Painter Studio", (40, 160), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, (209, 8, 105), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "text.png"), img_text)
    
    # 12. portrait_light.png: soft lighting portrait
    if os.path.exists("inputs/retrato.jpg"):
        shutil.copy("inputs/retrato.jpg", os.path.join(output_dir, "portrait_light.png"))
    else:
        img_port = np.full((300, 300, 3), (220, 215, 210), dtype=np.uint8)
        cv2.circle(img_port, (150, 150), 80, (180, 160, 210), -1)
        cv2.imwrite(os.path.join(output_dir, "portrait_light.png"), img_port)
        
    # 13. portrait_dark.png: chiaroscuro portrait
    img_dark = np.full((300, 300, 3), (20, 20, 20), dtype=np.uint8)
    cv2.circle(img_dark, (160, 140), 90, (40, 50, 80), -1)
    cv2.ellipse(img_dark, (170, 130), (50, 70), 20, 0, 360, (120, 140, 190), -1)
    cv2.circle(img_dark, (175, 120), 4, (255, 255, 255), -1) # specular catchlight
    cv2.imwrite(os.path.join(output_dir, "portrait_dark.png"), img_dark)
    
    # 14. portrait_skin_gradient.png: smooth skin tone transitions
    img_skin = np.zeros((250, 250, 3), dtype=np.uint8)
    for y in range(250):
        t = y / 250.0
        # Gradient from highlight skin to shadow skin
        b = int(140 * (1 - t) + 40 * t)
        g = int(180 * (1 - t) + 70 * t)
        r = int(240 * (1 - t) + 120 * t)
        img_skin[y, :] = (b, g, r)
    cv2.circle(img_skin, (125, 70), 10, (255, 255, 255), -1) # specular highlight
    cv2.imwrite(os.path.join(output_dir, "portrait_skin_gradient.png"), img_skin)
    
    # 15. hair.png: directional hair flow strokes
    img_hair = np.full((300, 300, 3), 255, dtype=np.uint8)
    for i in range(50, 250, 12):
        pts_hair = []
        for t in np.linspace(0, 1, 30):
            hx = int(i + 35 * np.sin(t * np.pi * 1.5))
            hy = int(40 + t * 220)
            pts_hair.append((hx, hy))
        pts_arr = np.array(pts_hair, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_hair, [pts_arr], False, (40, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "hair.png"), img_hair)
    
    # 16. gradient.png: continuous color spectrum gradient
    img_grad = np.zeros((200, 300, 3), dtype=np.uint8)
    for x in range(300):
        h_val = int((x / 300.0) * 179)
        hsv_pixel = np.uint8([[[h_val, 220, 240]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        img_grad[:, x] = bgr_pixel
    cv2.imwrite(os.path.join(output_dir, "gradient.png"), img_grad)
    
    # 17. low_contrast_photo.png: subtle low contrast scene
    img_low = np.full((250, 250, 3), (120, 125, 122), dtype=np.uint8)
    cv2.circle(img_low, (125, 125), 60, (132, 138, 135), -1)
    cv2.rectangle(img_low, (80, 80), (170, 170), (110, 115, 112), 2)
    cv2.imwrite(os.path.join(output_dir, "low_contrast_photo.png"), img_low)
    
    # 18. geometry.png: concentric shapes, sharp corners, and concave hole
    img_geom = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img_geom, (40, 40), (260, 260), (50, 180, 50), -1)
    cv2.circle(img_geom, (150, 150), 75, (255, 255, 255), -1) # inner hole
    cv2.rectangle(img_geom, (120, 120), (180, 180), (0, 0, 200), -1) # inner center block
    cv2.imwrite(os.path.join(output_dir, "geometry.png"), img_geom)
    
    files = os.listdir(output_dir)
    print(f"Dataset canônico gerado com sucesso em {output_dir}! Total de imagens: {len(files)}")
    return files

if __name__ == "__main__":
    generate_canonical_dataset()
