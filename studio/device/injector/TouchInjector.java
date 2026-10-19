package igp;

import android.os.SystemClock;
import android.view.InputDevice;
import android.view.InputEvent;
import android.view.MotionEvent;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;

/**
 * Resident touch injector run through app_process as the shell user.
 *
 * Protocol (one command per line on stdin, one reply per line on stdout):
 *   S <dt_ms> x0 y0 x1 y1 ...   one continuous stroke: DOWN, MOVEs paced dt_ms apart, UP
 *   T x y                        tap
 *   M <mode>                     injection mode: 0 async, 2 wait for finish (default)
 *   P                            ping
 *   Q                            quit
 * Every command is answered with "OK <n>" or "ERR <n> <message>".
 */
public final class TouchInjector {
    private static Object inputManager;
    private static Method injectMethod;
    private static int injectMode = 2;

    private static void init() throws Exception {
        Class<?> cls;
        try {
            cls = Class.forName("android.hardware.input.InputManagerGlobal");
        } catch (ClassNotFoundException e) {
            cls = Class.forName("android.hardware.input.InputManager");
        }
        inputManager = cls.getDeclaredMethod("getInstance").invoke(null);
        injectMethod = cls.getMethod("injectInputEvent", InputEvent.class, int.class);
    }

    private static boolean inject(long downTime, long eventTime, int action, float x, float y) throws Exception {
        MotionEvent.PointerProperties[] props = { new MotionEvent.PointerProperties() };
        props[0].id = 0;
        props[0].toolType = MotionEvent.TOOL_TYPE_FINGER;
        MotionEvent.PointerCoords[] coords = { new MotionEvent.PointerCoords() };
        coords[0].x = x;
        coords[0].y = y;
        coords[0].pressure = action == MotionEvent.ACTION_UP ? 0f : 1f;
        coords[0].size = 0.05f;
        MotionEvent event = MotionEvent.obtain(downTime, eventTime, action, 1, props, coords,
                0, 0, 1f, 1f, 0, 0, InputDevice.SOURCE_TOUCHSCREEN, 0);
        try {
            // Mode 2 = WAIT_FOR_FINISH keeps strict ordering with the app's input queue.
            return (Boolean) injectMethod.invoke(inputManager, event, injectMode);
        } finally {
            event.recycle();
        }
    }

    private static void sleepUntil(long uptimeMs) {
        long remaining = uptimeMs - SystemClock.uptimeMillis();
        if (remaining > 0) {
            SystemClock.sleep(remaining);
        }
    }

    private static void stroke(float dt, float[] xy) throws Exception {
        long down = SystemClock.uptimeMillis();
        inject(down, down, MotionEvent.ACTION_DOWN, xy[0], xy[1]);
        int n = xy.length / 2;
        for (int i = 1; i < n; i++) {
            long t = down + Math.round(i * dt);
            sleepUntil(t);
            inject(down, SystemClock.uptimeMillis(), MotionEvent.ACTION_MOVE, xy[2 * i], xy[2 * i + 1]);
        }
        sleepUntil(down + Math.round(n * dt));
        inject(down, SystemClock.uptimeMillis(), MotionEvent.ACTION_UP, xy[2 * n - 2], xy[2 * n - 1]);
    }

    public static void main(String[] args) throws Exception {
        init();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("READY");
        System.out.flush();
        long n = 0;
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            n++;
            String[] parts = line.split("\\s+");
            try {
                switch (parts[0]) {
                    case "S": {
                        float dt = Float.parseFloat(parts[1]);
                        float[] xy = new float[parts.length - 2];
                        if (xy.length < 2 || xy.length % 2 != 0) {
                            throw new IllegalArgumentException("odd coordinate count");
                        }
                        for (int i = 0; i < xy.length; i++) {
                            xy[i] = Float.parseFloat(parts[i + 2]);
                        }
                        stroke(dt, xy);
                        break;
                    }
                    case "T":
                        stroke(40, new float[] { Float.parseFloat(parts[1]), Float.parseFloat(parts[2]) });
                        break;
                    case "M":
                        injectMode = Integer.parseInt(parts[1]);
                        break;
                    case "P":
                        break;
                    case "Q":
                        System.out.println("OK " + n);
                        System.out.flush();
                        return;
                    default:
                        throw new IllegalArgumentException("unknown command " + parts[0]);
                }
                System.out.println("OK " + n);
            } catch (Exception e) {
                Throwable cause = e.getCause() != null ? e.getCause() : e;
                System.out.println("ERR " + n + " " + cause);
            }
            System.out.flush();
        }
    }
}
