# FF Machine
# Copyright (c) 2025 Oljake
# Licensed under the MIT License – see LICENSE file for details.

if __name__ == "__main__":
    import call_vote
    import threading
    import queue
    import use_model
    import time
    import torch
    import customtkinter as ctk
    import pickle
    from tkinter import messagebox
    import sys
    import os
    import atexit

    import overlay

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    running = False
    thread = None
    q = queue.Queue()
    stop_event = threading.Event()

    class_names = ['0', '1', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7', '8', '9', '-']

    def worker():

        global running
        while not stop_event.is_set():
            start = time.perf_counter()

            imgs = [use_model.transform(use_model.capture(r)).unsqueeze(0) for r in use_model.regions]

            if not imgs:
                insert_to_console(
                    "⚠️  **VALORANT** window not found.\n⚠️ Please launch the game on your **main monitor** using its **native resolution**, then restart the application.\n",
                    "warning")
                stop()
                return

            batch = torch.cat(imgs).to(use_model.device)

            with torch.no_grad():
                preds = use_model.model(batch).argmax(dim=1).tolist()

            use_model.detected_history.append(preds)

            per_region = list(zip(*use_model.detected_history))
            avg_detected = [use_model.Counter(region).most_common(1)[0][0] for region in per_region]

            def class_value(i):
                val = class_names[i]
                return int(val) if val.isdigit() else 0

            total = sum(class_value(i) for i in avg_detected)

            if total in [4, 12]:
                for i in range(sleep_after_combo_int, 0, -1):
                    insert_to_console(f"------------> Surrendering in {i}... <------------\n", "ligh_cyan")
                    msg = f"Surrendering in {i}"
                    overlay.overlay_q.put(msg)
                    for _ in range(10):
                        if stop_event.is_set():
                            break
                        time.sleep(0.1)
                    if stop_event.is_set():
                        break

                if stop_event.is_set():
                    break

                overlay.overlay_q.put("Surrendering")
                call_vote.force_focus_and_send_commands("VALORANT  ", ff_amount_int)

                insert_to_console(f"\n------------> Finished voting <------------\n", "process_color")

                use_model.detected_history.clear()
                stop()
                break

            time.sleep(max(0, use_model.target_dt - (time.perf_counter() - start)))

            if running:
                detected = ' : '.join(class_names[i] for i in preds)
                average = ' : '.join(class_names[i] for i in avg_detected)
                fps = f"{1 / (time.perf_counter() - start):.2f}"


                if n_avg_checkbox_var.get() and fps_checkbox_var.get():
                    q.put((f"▶ Detected: [{detected}]    ⟶ Average: [{average}]    ⏱ FPS: {fps}", "white"))
                    overlay.overlay_data = f"Det: {detected} Avg: {average} FPS: {fps}"


                elif n_avg_checkbox_var.get():
                    q.put((f"▶ Detected: [{detected}]    ⟶ Average: [{average}]", "white"))
                    overlay.overlay_data = f"Det: {detected} Avg: {average}"

                elif fps_checkbox_var.get():
                    q.put((f"▶ Detected: [{detected}]    ⏱ FPS: {fps}", "white"))
                    overlay.overlay_data = f"Det: {detected} FPS: {fps}"

                else:
                    q.put((f"▶ Detected: [{detected}]", "white"))
                    overlay.overlay_data = f"Det: {detected}"

                overlay.overlay_q.put(overlay.overlay_data)


    def start():
        global running, thread, stop_event
        insert_to_console(f"------------> Starting detection <------------\n\n", "process_color")

        running = True
        stop_event.clear()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        overlay.start_overlay()

        insert_to_console(f"------------> Detection started <------------\n\n\n", "process_color")

        start_btn.configure(state=ctk.DISABLED)


    def stop():
        global running, stop_event

        if not running:
            return

        insert_to_console("\n------------> Stopping . . . . . . . <------------\n\n", "process_color")

        running = False
        stop_event.set()
        overlay.stop_overlay()

        def wait_thread():
            if thread:
                thread.join()

            with q.mutex:
                q.queue.clear()

            start_btn.configure(state=ctk.NORMAL)

            insert_to_console(f"\n------------> Process stopped <------------\n\n\n", "process_color")

        threading.Thread(target=wait_thread, daemon=True).start()

    def insert_to_console(text, color="white"):
        console.configure(state="normal")
        console.insert(ctk.END, text, color)
        console.yview_moveto(1)
        console.configure(state="disabled")

    def update_console():
        last_visible = console.yview()
        at_bottom = last_visible[1] == 1.0

        while not q.empty():
            console.configure(state="normal")
            item = q.get()
            if isinstance(item, tuple):
                text, color = item
            else:
                text, color = item, None

            start_idx = console.index("end-1c")
            console.insert(ctk.END, text + '\n')
            end_idx = console.index("end-1c")

            if color:
                console.tag_add(color, start_idx, end_idx)

            console.configure(state="disabled")

        if at_bottom:
            console.yview_moveto(1)
        root.after(100, update_console)


    def clean_and_clamp(entry_var, min_val, max_val, default):
        val = entry_var.get()
        try:
            val = int(val)
        except ValueError:
            entry_var.set(str(default))
            return default

        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val

        entry_var.set(str(val))
        return val


    def apply_settings():
        if running:
            messagebox.showwarning("Warning", "Stop before applying settings.")
            return

        ff_amount = clean_and_clamp(ff_amount_var, 1, 999, 350)
        sleep_after_combo = clean_and_clamp(sleep_after_combo_var, 0, 7, 5)
        max_detected_history = clean_and_clamp(max_detected_history_var, 1, 999, 20)
        target_FPS = clean_and_clamp(target_fps_var, 10, 240, 30)

        n_avg_enabled = n_avg_checkbox_var.get()
        fps_enabled = fps_checkbox_var.get()

        try:
            with open("config.pkl", "rb") as f:
                full_config = pickle.load(f)
        except (FileNotFoundError, EOFError):
            full_config = {}

        partial_config = {
            "ff_amount": ff_amount,
            "sleep_after_combo": sleep_after_combo,
            "max_detected_history": max_detected_history,
            "target_FPS": target_FPS,
            "n_avg_checkbox_var": n_avg_enabled,
            "fps_checkbox_var": fps_enabled,
        }

        full_config.update(partial_config)

        with open("config.pkl", "wb") as f:
            pickle.dump(full_config, f)

        global ff_amount_int, sleep_after_combo_int, max_detected_history_int, target_fps_int
        ff_amount_int, sleep_after_combo_int, max_detected_history_int, target_fps_int = (
            ff_amount,
            sleep_after_combo,
            max_detected_history,
            target_FPS,
        )

        use_model.target_dt = 1 / target_FPS
        use_model.detected_history = use_model.deque(maxlen=max_detected_history)

        insert_to_console("------------> Settings applied <------------\n\n", "green")
        insert_to_console(f"----> Vote Attempts: {ff_amount}\n", "green")
        insert_to_console(f"----> Pre-vote Sleep: {sleep_after_combo}\n", "green")
        insert_to_console(f"----> N average: {max_detected_history} (Show: {n_avg_enabled})\n", "green")
        insert_to_console(f"----> Target FPS: {target_FPS} (Show: {fps_enabled})\n\n\n", "green")

    def validate_numeric(P):
        return P.isdigit() or P == ""


    root = ctk.CTk()
    root.geometry("911x424")
    root.title("FF Machine")
    root.resizable(False, False)

    def resource_path(relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)


    root.iconbitmap(resource_path("images/icon.ico"))

    main_frame = ctk.CTkFrame(root)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)

    controls_frame = ctk.CTkFrame(main_frame, width=200)
    controls_frame.pack(side="left", fill="y", padx=10, pady=10)

    console = ctk.CTkTextbox(main_frame, height=300, width=600, state="disabled", wrap="word")
    console.pack(side="right", fill="both", expand=True)

    console.configure(state="normal")

    # Before root.mainloop(), define your color tags:
    console.tag_config("white", foreground="#e8a246")
    console.tag_config("green", foreground="#1da34a")
    console.tag_config("cyan", foreground="#1c979c")
    console.tag_config("ligh_cyan", foreground="#21c5cc")
    console.tag_config("process_color", foreground="#a3279d")
    console.tag_config("warning", foreground="#e65353")



    # Display messages in prompt when starting the app
    welcome_message = ("------------> Welcome to the FF Machine, Version B • 0.92 <------------\n\n")

    start_up_message = (
        "How it works:\n"
        "• Click   **Start**   to begin monitoring the game score from your screen.\n\n"
        
        "• When a combined score of   **4**   or   **12**   is detected (based on average detection):\n"
        "    → FF Machine forces focus to the VALORANT window\n"
        "    → Sends   **/t /ff**   to initiate a team surrender (avoids party/private chat)\n"
        "    → Detection automatically stops after voting is triggered\n\n"
        
        "• Click   **Stop**   to halt score monitoring — but note:\n"
        "    → 'Stop' only ends detection, not the surrender process\n"
        "    → If voting has already started, it will continue to send the pre-set number of vote attempts\n\n"
        
        "Notes:\n"
        "• FF Machine does   **not**   return focus to your previous window after triggering the vote.\n"
        "• Please ensure Valorant is actively running and prominently displayed on your screen to ensure flawless operation.\n\n"
    )


    warning_message = (
        "⚠️ Use at   **your own risk**   — this may violate VALORANT’s terms of service. I am   **not responsible**   \n"
        "⚠️ for any issues or consequences resulting from its use.\n\n"
    )

    warning_message_cuda = ("⚠️  **CUDA**   is not available. Running on   **CPU**  , which may be slower.\n\n")


    insert_to_console(welcome_message, "green")
    insert_to_console(start_up_message, "green")
    insert_to_console(warning_message, "warning")

    start_btn = ctk.CTkButton(controls_frame, text="Start", command=start)
    start_btn.pack(padx=10, pady=10)
    stop_btn = ctk.CTkButton(controls_frame, text="Stop", command=stop)
    stop_btn.pack(padx=10, pady=10)

    with open("config.pkl", "rb") as f:
        loaded_config = pickle.load(f)

    run_model_path = ctk.StringVar(value=str(loaded_config.get("run_model_path", "models/digit_cnn.pth")))
    num_classes = ctk.StringVar(value=str(loaded_config.get("num_classes", 15)))

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -  # - # - # - # -

    cuda_failed = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            insert_to_console(f"⚠️ CUDA error: {e}", "error")
            device = torch.device("cpu")
            cuda_failed = True

    if device.type == "cpu" and not cuda_failed:
        insert_to_console(warning_message_cuda, "warning")

    model = use_model.DigitCNN(int(num_classes.get())).to(device)
    model.load_state_dict(torch.load(run_model_path.get(), map_location=device))
    model.eval()

    use_model.target_dt = 1 / loaded_config.get("target_FPS", 30)
    use_model.regions = use_model.init()


    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

    atexit.register(overlay.stop_overlay)

    n_avg_checkbox_var = ctk.BooleanVar()
    fps_checkbox_var = ctk.BooleanVar()

    def load_config():
        try:
            loaded = loaded_config  # preloaded dict
            ff_amount_var = ctk.StringVar(value=str(loaded.get("ff_amount", 20)))
            sleep_after_combo_var = ctk.StringVar(value=str(loaded.get("sleep_after_combo", 5)))
            max_detected_history_var = ctk.StringVar(value=str(loaded.get("max_detected_history", 20)))
            target_fps_var = ctk.StringVar(value=str(loaded.get("target_FPS", 30)))

            ff_amount_int = int(loaded.get("ff_amount", 20))
            sleep_after_combo_int = int(loaded.get("sleep_after_combo", 5))
            max_detected_history_int = int(loaded.get("max_detected_history", 20))
            target_fps_int = int(loaded.get("target_FPS", 30))

            n_avg_checkbox_var.set(bool(loaded.get("n_avg_checkbox_var", True)))
            fps_checkbox_var.set(bool(loaded.get("fps_checkbox_var", True)))


        except Exception:
            ff_amount_var = ctk.StringVar(value="20")
            sleep_after_combo_var = ctk.StringVar(value="5")
            max_detected_history_var = ctk.StringVar(value="20")
            target_fps_var = ctk.StringVar(value="30")

            ff_amount_int, sleep_after_combo_int, max_detected_history_int, target_fps_int = 20, 5, 20, 30

            n_avg_checkbox_var.set(True)
            fps_checkbox_var.set(True)

        return (
            ff_amount_var,
            sleep_after_combo_var,
            max_detected_history_var,
            target_fps_var,
            ff_amount_int,
            sleep_after_combo_int,
            max_detected_history_int,
            target_fps_int,
            n_avg_checkbox_var,
            fps_checkbox_var
        )

    ff_amount_var, sleep_after_combo_var, max_detected_history_var, target_fps_var, ff_amount_int, \
        sleep_after_combo_int, max_detected_history_int, target_fps_int, n_avg_checkbox_var, fps_checkbox_var = load_config()



    validate_int = (root.register(validate_numeric), '%P')
    frame_color = controls_frame.cget("fg_color")  # Get parent bg color


    ctk.CTkLabel(controls_frame, text="Vote Attempts").pack()
    ctk.CTkEntry(controls_frame, textvariable=ff_amount_var, validate="key", validatecommand=validate_int).pack(padx=(10, 10), pady=2)

    ctk.CTkLabel(controls_frame, text="Pre-vote Sleep").pack()
    ctk.CTkEntry(controls_frame, textvariable=sleep_after_combo_var, validate="key", validatecommand=validate_int).pack(padx=(10, 10), pady=2)

    # N average row
    ctk.CTkLabel(controls_frame, text="N average").pack()
    n_avg_row = ctk.CTkFrame(controls_frame, fg_color=frame_color)
    n_avg_row.pack(padx=10, pady=2)
    ctk.CTkEntry(n_avg_row, width=110, textvariable=max_detected_history_var, validate="key", validatecommand=validate_int).pack(side="left", padx=(10, 0))
    ctk.CTkCheckBox(n_avg_row, text="", width=20, variable=n_avg_checkbox_var).pack(side="left", padx=(5, 0))

    # Target FPS row
    ctk.CTkLabel(controls_frame, text="Target FPS").pack()
    fps_row = ctk.CTkFrame(controls_frame, fg_color=frame_color)
    fps_row.pack(padx=10, pady=2)
    ctk.CTkEntry(fps_row, width=110, textvariable=target_fps_var, validate="key", validatecommand=validate_int).pack(side="left", padx=(10, 0))
    ctk.CTkCheckBox(fps_row, text="", width=20, variable=fps_checkbox_var).pack(side="left", padx=(5, 0))


    ctk.CTkButton(controls_frame, text="Apply Settings", command=apply_settings).pack(padx=(10, 10), pady=10)

    start_btn.configure(state=ctk.NORMAL)
    root.after(100, update_console)
    root.mainloop()
