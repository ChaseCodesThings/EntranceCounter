import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import csv
import time

#Welcome to our code
# --- CONFIGURATION ---
PAGE_TITLE = "Occupancy Monitor"
CSV_FILE = "attendance_log.csv"
COOLDOWN_SECONDS = 3.0  # No double counts within 3 seconds

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- 1. STATE MANAGEMENT ---
if 'cam_index' not in st.session_state:
    st.session_state['cam_index'] = 0
if 'count_in' not in st.session_state:
    st.session_state['count_in'] = 0
if 'count_out' not in st.session_state:
    st.session_state['count_out'] = 0
if 'recent_events' not in st.session_state:
    st.session_state['recent_events'] = []
if 'tracker_state' not in st.session_state:
    st.session_state['tracker_state'] = {}

# Cooldown Timers {tracker_id: timestamp}
if 'last_entry_time' not in st.session_state:
    st.session_state['last_entry_time'] = {}
if 'last_exit_time' not in st.session_state:
    st.session_state['last_exit_time'] = {}


def cycle_camera():
    st.session_state['cam_index'] = (st.session_state['cam_index'] + 1) % 3


# --- 2. CORE FUNCTIONS ---

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')


def get_feet_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def log_event(event_type, tracker_id):
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Event", "ID"])
        writer.writerow([date_str, time_str, event_type, tracker_id])

    return {"Time": time_str, "Event": event_type, "ID": f"#{tracker_id}"}


# --- 3. UI LAYOUT ---

st.title(PAGE_TITLE)

with st.sidebar:
    st.header("Control Panel")

    st.subheader("Video Source")
    st.write(f"Current Input: Index {st.session_state['cam_index']}")
    st.button("Change Video Source", on_click=cycle_camera, use_container_width=True)

    st.divider()

    st.subheader("Calibration")
    st.caption("Move the zone up or down. The gap is fixed.")

    # --- UPDATED SLIDER LOGIC ---
    # We limit max to 0.90 so the bottom line (which is +0.10) doesn't go off screen
    line_door_pct = st.slider("Zone Position (Top Line)", 0.0, 0.90, 0.40, 0.01)

    # Automatically calculate the bottom line with a 0.10 gap
    line_room_pct = line_door_pct + 0.10

    # Visual feedback removed per request
    # ----------------------------

    st.divider()

    st.subheader("System Status")
    run_ai = st.toggle("Active Monitoring", value=False)

    st.divider()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("Export CSV Log", f, file_name="attendance_log.csv", use_container_width=True)

# Main Grid
col_video, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("Real-Time Metrics")
    m1, m2 = st.columns(2)
    # PLACEHOLDERS for instant updates
    in_metric = m1.empty()
    out_metric = m2.empty()
    in_metric.metric("Total Entered", st.session_state['count_in'])
    out_metric.metric("Total Exited", st.session_state['count_out'])

    st.divider()
    st.subheader("Activity Log")
    log_table = st.empty()
    if st.session_state['recent_events']:
        df = pd.DataFrame(st.session_state['recent_events'][:8])
        log_table.dataframe(df, hide_index=True, use_container_width=True)

with col_video:
    video_placeholder = st.empty()


# --- 4. MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(st.session_state['cam_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        video_placeholder.error(f"Error: Video Source {st.session_state['cam_index']} unavailable.")
        return

    model = load_model()

    # Annotators
    tracker = sv.ByteTrack()
    box_an = sv.BoxAnnotator(thickness=3)  # Thicker for presentation
    label_an = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)

    while True:
        ret, frame = cap.read()
        if not ret:
            video_placeholder.error("Video stream disconnected.")
            break

        h, w = frame.shape[:2]

        # CALCULATE LINES (Updated to use the auto-calculated variable)
        door_y = int(h * line_door_pct)
        room_y = int(h * line_room_pct)

        # Draw Lines (Thicker for visibility)
        cv2.line(frame, (0, door_y), (w, door_y), (255, 255, 255), 3)
        cv2.line(frame, (0, room_y), (w, room_y), (0, 255, 0), 3)

        # Add Labels to the Video Feed
        cv2.putText(frame, "DOOR ZONE", (10, door_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ROOM ZONE", (10, room_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if run_ai:
            results = model(frame, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(results)
            dets = tracker.update_with_detections(dets)

            current_states = {}
            current_time = time.time()

            for i, (tid, cid) in enumerate(zip(dets.tracker_id, dets.class_id)):
                # GUARD CLAUSE: Only Humans
                if cid != 0: continue

                bbox = dets.xyxy[i]
                _, feet_y = get_feet_position(bbox)

                # Determine Zone
                if feet_y < door_y:
                    curr_pos = "OUTSIDE"
                elif feet_y < room_y:
                    curr_pos = "BETWEEN"
                else:
                    curr_pos = "INSIDE"

                current_states[tid] = curr_pos

                # LOGIC
                prev_pos = st.session_state['tracker_state'].get(tid)

                if prev_pos:
                    # ENTRY LOGIC
                    if (prev_pos == "OUTSIDE" or prev_pos == "BETWEEN") and curr_pos == "INSIDE":
                        # CHECK COOLDOWN
                        last_time = st.session_state['last_entry_time'].get(tid, 0)
                        if current_time - last_time > COOLDOWN_SECONDS:
                            st.session_state['count_in'] += 1
                            st.session_state['recent_events'].insert(0, log_event("ENTRY", tid))
                            st.session_state['last_entry_time'][tid] = current_time  # Set Timer

                            # Update UI
                            in_metric.metric("Total Entered", st.session_state['count_in'])
                            df = pd.DataFrame(st.session_state['recent_events'][:8])
                            log_table.dataframe(df, hide_index=True, use_container_width=True)

                    # EXIT LOGIC
                    elif (prev_pos == "INSIDE" or prev_pos == "BETWEEN") and curr_pos == "OUTSIDE":
                        # CHECK COOLDOWN
                        last_time = st.session_state['last_exit_time'].get(tid, 0)
                        if current_time - last_time > COOLDOWN_SECONDS:
                            st.session_state['count_out'] += 1
                            st.session_state['recent_events'].insert(0, log_event("EXIT", tid))
                            st.session_state['last_exit_time'][tid] = current_time  # Set Timer

                            # Update UI
                            out_metric.metric("Total Exited", st.session_state['count_out'])
                            df = pd.DataFrame(st.session_state['recent_events'][:8])
                            log_table.dataframe(df, hide_index=True, use_container_width=True)

            # Update Memory
            st.session_state['tracker_state'].update(current_states)

            # Visuals
            labels = []
            for tid, cid in zip(dets.tracker_id, dets.class_id):
                object_name = model.names[cid]
                if cid == 0:
                    labels.append(f"#{tid} Person")
                else:
                    labels.append(f"{object_name}")

            frame = box_an.annotate(frame, dets)
            frame = label_an.annotate(frame, dets, labels=labels)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


if __name__ == "__main__":
    main()