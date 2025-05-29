import threading
import queue
import pygame
import win32con
import win32gui
import ctypes

overlay_q = queue.Queue()
overlay_thread = None
overlay_running = threading.Event()

WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
GWL_EXSTYLE = -20


def make_clickthrough(hwnd):
    ex_style = win32gui.GetWindowLong(hwnd, GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    # SetLayeredWindowAttributes with alpha=255 and color key=black (0,0,0) for transparency
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 255, 0x1)


def get_primary_monitor_size():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def hide_from_taskbar_and_alt_tab(hwnd):
    ex_style = win32gui.GetWindowLong(hwnd, GWL_EXSTYLE)
    # Remove WS_EX_APPWINDOW and add WS_EX_TOOLWINDOW to hide from taskbar and Alt+Tab
    ex_style = (ex_style & ~win32con.WS_EX_APPWINDOW) | win32con.WS_EX_TOOLWINDOW
    win32gui.SetWindowLong(hwnd, GWL_EXSTYLE, ex_style)


def is_correct_combination(data):
    return "correct_combo" in data


def overlay_worker(q, flag):
    pygame.init()
    w, h = 280, 38
    screen = pygame.display.set_mode((w, h), pygame.NOFRAME | pygame.SRCALPHA)
    hwnd = pygame.display.get_wm_info()['window']
    hide_from_taskbar_and_alt_tab(hwnd)

    x, y = 10, 20
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, w, h, win32con.SWP_SHOWWINDOW)
    make_clickthrough(hwnd)

    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    data = ""
    shown = False

    while flag.is_set():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                flag.clear()

        try:
            while 1:
                data = q.get_nowait()
        except queue.Empty:
            pass

        if not shown and is_correct_combination(data):
            data = "surrendering"
            shown = True

        screen.fill((0, 0, 0, 0))  # fully transparent

        if data:
            text_surf = font.render(data, True, (0, 255, 255))
            screen.blit(text_surf, (10, 10))

        pygame.display.update()
        clock.tick(60)

        if shown:
            pygame.time.wait(2000)
            flag.clear()

    pygame.quit()

def start_overlay():
    global overlay_thread
    overlay_running.set()
    overlay_thread = threading.Thread(target=overlay_worker, args=(overlay_q, overlay_running), daemon=True)
    overlay_thread.start()


def stop_overlay():
    overlay_running.clear()
    if overlay_thread:
        overlay_thread.join()
