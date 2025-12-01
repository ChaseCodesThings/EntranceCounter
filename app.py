import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import csv
#dummychange
# --- CONFIGURATION ---
PAGE_TITLE = "Occupancy Monitor (Secure)"
CSV_FILE = "attendance_log.csv"

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
# Tracker State: {tracker_id: "OUTSIDE" | "BETWEEN" | "INSIDE"}
if 'tracker_state' not in st.session_state:
    st.session_state['tracker_state'] = {}


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
    st.write(f"Index: {st.session_state['cam_index']}")
    st.button("Switch Camera", on_click=cycle_camera, use_container_width=True)

    st.divider()

    st.subheader("Calibration")
    st.caption("Adjust lines to floor perspective.")
    line_door_pct = st.slider("Door Line (White)", 0.0, 1.0, 0.50, 0.01)
    line_room_pct = st.slider("Room Line (Green)", 0.0, 1.0, 0.85, 0.01)

    st.divider()

    st.subheader("System Status")
    run_ai = st.toggle("Active Monitoring", value=False)

    st.divider()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("Export Log", f, file_name="attendance_log.csv", use_container_width=True)

col_video, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("Real-Time Metrics")
    m1, m2 = st.columns(2)
    # PLACEHOLDERS for zero-flicker updates
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

    # Tracker Config (Updated for new Supervision version)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )

    box_an = sv.BoxAnnotator(thickness=2)
    label_an = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)
    trace_an = sv.TraceAnnotator(thickness=2, trace_length=30)  # Professional path tracing

    while True:
        ret, frame = cap.read()
        if not ret:
            video_placeholder.error("Video stream disconnected.")
            break

        h, w = frame.shape[:2]
        door_y = int(h * line_door_pct)
        room_y = int(h * line_room_pct)

        # Draw Lines
        cv2.line(frame, (0, door_y), (w, door_y), (255, 255, 255), 2)
        cv2.line(frame, (0, room_y), (w, room_y), (0, 255, 0), 2)

        # Clean Labels
        cv2.putText(frame, "OUTSIDE", (10, door_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "INSIDE", (10, room_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if run_ai:
            results = model(frame, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(results)
            dets = tracker.update_with_detections(dets)

            current_states = {}

            for i, (tid, cid) in enumerate(zip(dets.tracker_id, dets.class_id)):
                if cid != 0: continue

                bbox = dets.xyxy[i]
                _, feet_y = get_feet_position(bbox)

                # Zone Logic
                if feet_y < door_y:
                    curr_pos = "OUTSIDE"
                elif feet_y < room_y:
                    curr_pos = "BETWEEN"
                else:
                    curr_pos = "INSIDE"

                current_states[tid] = curr_pos

                # Check Transitions
                prev_pos = st.session_state['tracker_state'].get(tid)

                if prev_pos:
                    # ENTRY
                    if (prev_pos == "OUTSIDE" or prev_pos == "BETWEEN") and curr_pos == "INSIDE":
                        st.session_state['count_in'] += 1
                        st.session_state['recent_events'].insert(0, log_event("ENTRY", tid))
                        in_metric.metric("Total Entered", st.session_state['count_in'])
                        df = pd.DataFrame(st.session_state['recent_events'][:8])
                        log_table.dataframe(df, hide_index=True, use_container_width=True)
                        st.session_state['tracker_state'][tid] = "INSIDE"

                    # EXIT
                    elif (prev_pos == "INSIDE" or prev_pos == "BETWEEN") and curr_pos == "OUTSIDE":
                        st.session_state['count_out'] += 1
                        st.session_state['recent_events'].insert(0, log_event("EXIT", tid))
                        out_metric.metric("Total Exited", st.session_state['count_out'])
                        df = pd.DataFrame(st.session_state['recent_events'][:8])
                        log_table.dataframe(df, hide_index=True, use_container_width=True)
                        st.session_state['tracker_state'][tid] = "OUTSIDE"

            st.session_state['tracker_state'].update(current_states)

            # Visuals
            labels = []
            for tid, cid in zip(dets.tracker_id, dets.class_id):
                object_name = model.names[cid]
                if cid == 0:
                    labels.append(f"#{tid} Person")
                else:
                    labels.append(f"{object_name}")

            frame = trace_an.annotate(frame, dets)
            frame = box_an.annotate(frame, dets)
            frame = label_an.annotate(frame, dets, labels=labels)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


if __name__ == "__main__":
    main()