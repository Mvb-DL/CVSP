#!/usr/bin/env python3
"""
YOLOv8 Multi-Object Tracking (MOT) - Live Viewer
Zeigt Video mit Tracking-IDs und umfangreichen Metriken in Echtzeit
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np
from collections import defaultdict


class TrackingMetrics:
    """Verwaltet Tracking-spezifische Metriken"""
    
    def __init__(self):
        self.frame_count = 0
        self.inference_times = []
        self.start_time = time.time()
        
        # Tracking-spezifisch
        self.active_tracks = set()  # Aktuell sichtbare Track-IDs
        self.all_tracks_ever = set()  # Alle jemals gesehenen IDs
        self.track_lifetimes = defaultdict(int)  # Wie lange jede ID sichtbar war
        self.track_first_seen = {}  # Wann wurde ID das erste Mal gesehen
        self.track_last_seen = {}  # Wann wurde ID zuletzt gesehen
        self.detections_per_frame = []
        
    def update(self, track_ids, inference_time):
        """Update mit aktuellen Track-IDs"""
        self.frame_count += 1
        self.inference_times.append(inference_time)
        
        # Nur letzte 30 Frames
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        # Tracking-Updates
        current_ids = set(track_ids)
        self.active_tracks = current_ids
        self.all_tracks_ever.update(current_ids)
        self.detections_per_frame.append(len(current_ids))
        
        # Lifetime tracking
        for tid in current_ids:
            self.track_lifetimes[tid] += 1
            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = self.frame_count
            self.track_last_seen[tid] = self.frame_count
        
        if len(self.detections_per_frame) > 30:
            self.detections_per_frame.pop(0)
    
    def get_avg_inference_time(self):
        return np.mean(self.inference_times) if self.inference_times else 0
    
    def get_fps(self):
        avg_time = self.get_avg_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def get_avg_detections(self):
        return np.mean(self.detections_per_frame) if self.detections_per_frame else 0
    
    def get_total_unique_tracks(self):
        return len(self.all_tracks_ever)
    
    def get_active_tracks_count(self):
        return len(self.active_tracks)
    
    def get_longest_track(self):
        if not self.track_lifetimes:
            return 0, 0
        tid = max(self.track_lifetimes, key=self.track_lifetimes.get)
        return tid, self.track_lifetimes[tid]
    
    def get_elapsed_time(self):
        return time.time() - self.start_time


def draw_enhanced_metrics_panel(frame, metrics, total_frames, tracker_name):
    """Zeichnet erweitertes Metriken-Panel mit Tracking-Info"""
    
    h, w = frame.shape[:2]
    panel_height = 280
    panel_width = 480
    
    # Semi-transparentes Panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Rahmen (Grün für Tracking aktiv)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), 
                  (0, 255, 0), 2)
    
    # Metriken Text
    y_offset = 40
    line_height = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    
    # Titel
    cv2.putText(frame, "=== MULTI-OBJECT TRACKING ===", (20, y_offset),
                font, 0.65, (0, 255, 255), thickness)
    y_offset += 35
    
    # Frame Info
    progress = (metrics.frame_count / total_frames) * 100 if total_frames > 0 else 0
    cv2.putText(frame, f"Frame: {metrics.frame_count}/{total_frames} ({progress:.1f}%)",
                (20, y_offset), font, 0.55, color, 1)
    y_offset += line_height
    
    # === TRACKING METRIKEN ===
    
    # Aktive Tracks (HERVORGEHOBEN)
    active_count = metrics.get_active_tracks_count()
    track_color = (0, 255, 255) if active_count > 0 else (100, 100, 100)
    cv2.putText(frame, f"Aktive Tracks: {active_count}",
                (20, y_offset), font, font_scale, track_color, thickness)
    y_offset += line_height
    
    # Gesamt eindeutige Personen
    total_unique = metrics.get_total_unique_tracks()
    cv2.putText(frame, f"Gesamt (unique): {total_unique} Personen",
                (20, y_offset), font, 0.55, (150, 255, 150), 1)
    y_offset += line_height
    
    # Durchschnittliche Detektionen
    avg_det = metrics.get_avg_detections()
    cv2.putText(frame, f"Durchschn. Tracks/Frame: {avg_det:.1f}",
                (20, y_offset), font, 0.5, color, 1)
    y_offset += line_height
    
    # Längster Track
    longest_id, longest_frames = metrics.get_longest_track()
    if longest_frames > 0:
        cv2.putText(frame, f"Langster Track: ID#{longest_id} ({longest_frames} frames)",
                    (20, y_offset), font, 0.5, (255, 200, 100), 1)
    y_offset += line_height + 5
    
    # === PERFORMANCE ===
    fps = metrics.get_fps()
    inf_time = metrics.get_avg_inference_time() * 1000
    fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 8 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f} | Inferenz: {inf_time:.1f}ms",
                (20, y_offset), font, 0.55, fps_color, thickness)
    y_offset += line_height
    
    # Laufzeit
    elapsed = metrics.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    cv2.putText(frame, f"Laufzeit: {minutes:02d}:{seconds:02d}",
                (20, y_offset), font, 0.55, color, 1)
    y_offset += line_height
    
    # Tracker
    cv2.putText(frame, f"Tracker: {tracker_name}",
                (20, y_offset), font, 0.5, (200, 200, 200), 1)
    
    # Steuerung
    y_offset += 35
    cv2.putText(frame, "Q: Beenden | SPACE: Pause | S: Screenshot",
                (20, y_offset), font, 0.45, (150, 150, 150), 1)
    
    return frame


def draw_track_list(frame, active_ids, track_lifetimes, max_display=10):
    """Zeichnet Liste der aktiven Track-IDs rechts im Bild"""
    
    if not active_ids:
        return frame
    
    h, w = frame.shape[:2]
    
    # Panel rechts oben
    panel_width = 220
    panel_x = w - panel_width - 10
    panel_y = 10
    line_height = 25
    
    # Sortiere IDs nach Lifetime (längste zuerst)
    sorted_ids = sorted(active_ids, key=lambda x: track_lifetimes.get(x, 0), reverse=True)
    display_ids = sorted_ids[:max_display]
    
    panel_height = 50 + len(display_ids) * line_height
    
    # Semi-transparentes Panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Rahmen
    cv2.rectangle(frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (255, 255, 0), 2)
    
    # Titel
    cv2.putText(frame, "AKTIVE TRACKS", (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Track-Liste
    y = panel_y + 50
    for i, tid in enumerate(display_ids):
        lifetime = track_lifetimes.get(tid, 0)
        text = f"ID #{tid:3d} ({lifetime:3d}f)"
        
        # Farbe basierend auf Lifetime
        if lifetime > 100:
            color = (0, 255, 0)  # Grün = lange sichtbar
        elif lifetime > 30:
            color = (0, 255, 255)  # Gelb
        else:
            color = (255, 255, 255)  # Weiß = neu
        
        cv2.putText(frame, text, (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += line_height
    
    # Falls mehr IDs als angezeigt
    if len(sorted_ids) > max_display:
        cv2.putText(frame, f"... +{len(sorted_ids) - max_display} mehr", 
                   (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 MOT Live Viewer")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Pfad zum Modell (.pt)")
    parser.add_argument("--video", type=str, required=True,
                        help="Pfad zum Video")
    parser.add_argument("--tracker", type=str, default="botsort.yaml",
                        help="Tracker config (botsort.yaml oder bytetrack.yaml)")
    parser.add_argument("--conf", type=float, default=0.28,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.55,
                        help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=896,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (0 für cuda:0, cpu für CPU)")
    parser.add_argument("--save-video", action="store_true",
                        help="Speichere annotiertes Video")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Modell laden
    print(f"\n{'='*80}")
    print(f"🎯 MULTI-OBJECT TRACKING (MOT)")
    print(f"{'='*80}")
    print(f"🔍 Modell: {args.model}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Tracker: {args.tracker}")
    
    model = YOLO(args.model)
    
    # Video öffnen
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video nicht gefunden: {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Fehler beim Öffnen: {video_path}")
        return
    
    # Video-Eigenschaften
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {video_path.name}")
    print(f"   {width}x{height} @ {fps} FPS | {total_frames} frames")
    print(f"⚙️  Settings: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print(f"{'='*80}\n")
    
    # Output-Video
    out_writer = None
    if args.save_video:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_tracked.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Output: {output_path}\n")
    
    # Metriken
    metrics = TrackingMetrics()
    paused = False
    screenshot_count = 0
    
    # Fenster
    window_name = 'YOLOv8 Multi-Object Tracking - Live'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Tracker-Name für Display
    tracker_name = Path(args.tracker).stem.upper()
    
    print("🚀 Tracking gestartet!")
    print("   SPACE = Pause | Q = Beenden | S = Screenshot\n")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO TRACKING
                t_start = time.time()
                results = model.track(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    persist=True,  # WICHTIG für ID-Persistenz!
                    tracker=args.tracker,
                    classes=[0],  # Nur Person
                    verbose=False
                )
                inference_time = time.time() - t_start
                
                # Annotierter Frame
                annotated_frame = results[0].plot()
                
                # Track-IDs extrahieren
                track_ids = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    track_ids = [int(tid) for tid in results[0].boxes.id.cpu().numpy()]
                
                # Metriken updaten
                metrics.update(track_ids, inference_time)
                
                # Metriken-Panel zeichnen
                annotated_frame = draw_enhanced_metrics_panel(
                    annotated_frame,
                    metrics,
                    total_frames,
                    tracker_name
                )
                
                # Track-Liste zeichnen
                annotated_frame = draw_track_list(
                    annotated_frame,
                    metrics.active_tracks,
                    metrics.track_lifetimes
                )
                
                # Speichern
                if out_writer:
                    out_writer.write(annotated_frame)
                
                # Anzeigen
                cv2.imshow(window_name, annotated_frame)
            
            # Tastatur
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n⚠️  Beendet durch Benutzer")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"   {'PAUSIERT' if paused else 'FORTGESETZT'}")
            elif key == ord('s'):
                # Screenshot
                screenshot_count += 1
                screenshot_path = video_path.parent / f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(str(screenshot_path), annotated_frame)
                print(f"   📸 Screenshot gespeichert: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Abbruch (Ctrl+C)")
    
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        # FINALE STATISTIKEN
        print(f"\n{'='*80}")
        print(f"✅ TRACKING ZUSAMMENFASSUNG")
        print(f"{'='*80}")
        print(f"📊 TRACKING STATISTIKEN:")
        print(f"   Verarbeitete Frames:       {metrics.frame_count} / {total_frames}")
        print(f"   Eindeutige Personen:       {metrics.get_total_unique_tracks()}")
        print(f"   Durchschn. Tracks/Frame:   {metrics.get_avg_detections():.2f}")
        
        longest_id, longest_frames = metrics.get_longest_track()
        if longest_frames > 0:
            print(f"   Längster Track:            ID#{longest_id} ({longest_frames} frames)")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Durchschnittliche FPS:     {metrics.get_fps():.1f}")
        print(f"   Durchschn. Inferenz:       {metrics.get_avg_inference_time()*1000:.1f}ms")
        
        elapsed = metrics.get_elapsed_time()
        print(f"   Gesamtlaufzeit:            {int(elapsed//60)}:{int(elapsed%60):02d} min")
        
        if out_writer:
            print(f"\n💾 Output:                    {output_path}")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()