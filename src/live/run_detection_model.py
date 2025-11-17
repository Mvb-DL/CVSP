#!/usr/bin/env python3
"""
YOLOv8 SAR Model - Live Video Inference mit Metriken + Area-Filter
Erkennt Personen in Videos mit Ihrem trainierten Search & Rescue Modell
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np


class MetricsTracker:
    """Verwaltet und berechnet Metriken während der Inferenz"""

    def __init__(self):
        self.frame_count = 0
        self.total_detections = 0
        self.inference_times = []
        self.detections_per_frame = []
        self.start_time = time.time()

    def update(self, num_detections, inference_time):
        self.frame_count += 1
        self.total_detections += num_detections
        self.inference_times.append(inference_time)
        self.detections_per_frame.append(num_detections)

        # Nur letzte 30 Frames für gleitenden Durchschnitt
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        if len(self.detections_per_frame) > 30:
            self.detections_per_frame.pop(0)

    def get_avg_inference_time(self):
        return np.mean(self.inference_times) if self.inference_times else 0

    def get_fps(self):
        avg_time = self.get_avg_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0

    def get_avg_detections(self):
        return np.mean(self.detections_per_frame) if self.detections_per_frame else 0

    def get_max_detections(self):
        return max(self.detections_per_frame) if self.detections_per_frame else 0

    def get_min_detections(self):
        return min(self.detections_per_frame) if self.detections_per_frame else 0

    def get_elapsed_time(self):
        return time.time() - self.start_time


def draw_metrics_panel(frame, metrics, total_frames, current_detections, conf_threshold, min_area_ratio):
    """Zeichnet ein detailliertes Metriken-Panel auf den Frame"""

    h, w = frame.shape[:2]
    panel_height = 240
    panel_width = 450

    # Semi-transparentes Panel erstellen
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Rahmen
    cv2.rectangle(
        frame,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (0, 255, 0),
        2,
    )

    # Metriken Text
    y_offset = 45
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    # Titel
    cv2.putText(
        frame,
        "=== LIVE METRIKEN ===",
        (20, 35),
        font,
        0.7,
        (0, 255, 0),
        thickness,
    )

    # Frame Info
    progress = (metrics.frame_count / total_frames) * 100 if total_frames > 0 else 0
    cv2.putText(
        frame,
        f"Frame: {metrics.frame_count}/{total_frames} ({progress:.1f}%)",
        (20, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )
    y_offset += line_height

    # Aktuelle Detektionen (hervorgehoben)
    det_color = (0, 255, 255) if current_detections > 0 else (100, 100, 100)
    cv2.putText(
        frame,
        f"Personen (aktuell): {current_detections}",
        (20, y_offset),
        font,
        font_scale,
        det_color,
        thickness,
    )
    y_offset += line_height

    # Detektions-Statistiken
    cv2.putText(
        frame,
        f"Gesamt: {metrics.total_detections} | "
        f"Avg: {metrics.get_avg_detections():.1f} | "
        f"Max: {metrics.get_max_detections()}",
        (20, y_offset),
        font,
        0.5,
        color,
        1,
    )
    y_offset += line_height

    # Performance
    fps = metrics.get_fps()
    inf_time = metrics.get_avg_inference_time() * 1000  # in ms
    fps_color = (
        (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 8 else (0, 0, 255)
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Inferenz: {inf_time:.1f}ms",
        (20, y_offset),
        font,
        font_scale,
        fps_color,
        thickness,
    )
    y_offset += line_height

    # Laufzeit
    elapsed = metrics.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    cv2.putText(
        frame,
        f"Laufzeit: {minutes:02d}:{seconds:02d}",
        (20, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )
    y_offset += line_height

    # Confidence Threshold
    cv2.putText(
        frame,
        f"Confidence: {conf_threshold:.2f}",
        (20, y_offset),
        font,
        font_scale,
        (200, 200, 200),
        thickness,
    )
    y_offset += line_height

    # Min Area Ratio
    cv2.putText(
        frame,
        f"Min Area: {min_area_ratio * 100:.3f}% of frame",
        (20, y_offset),
        font,
        0.5,
        (180, 180, 180),
        1,
    )

    # Steuerungshinweise
    y_offset += 40
    cv2.putText(
        frame,
        "Q: Beenden | SPACE: Pause",
        (20, y_offset),
        font,
        0.5,
        (150, 150, 150),
        1,
    )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 SAR Model - Live Video Inference"
    )

    # Pflichtargumente
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pfad zum Modell (z.B. checkpoints/best.pt)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Pfad zum Input-Video",
    )

    # Optionale Argumente
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Pfad zum Output-Video (optional)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (Standard: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold für NMS (Standard: 0.5)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Inference image size (Standard: 800)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, Standard: auto)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=150,
        help="Maximale Anzahl Detektionen pro Bild (Standard: 150)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Speichere kein Output-Video",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0002,
        help="Minimale Boxfläche relativ zur Bildfläche (z.B. 0.0002 = 0.02%)",
    )

    args = parser.parse_args()

    # Modell laden
    print(f"\n{'='*80}")
    print(f"🔍 Lade Modell: {args.model}")
    model = YOLO(args.model)

    # Device auswählen
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # Video öffnen
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video nicht gefunden: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Fehler beim Öffnen des Videos: {video_path}")
        return

    # Video-Eigenschaften
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {video_path.name}")
    print(f"   Resolution: {width}x{height} @ {fps} FPS")
    print(f"   Frames: {total_frames}")
    print(
        f"⚙️  Inference Settings: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, "
        f"min_area_ratio={args.min_area_ratio}"
    )
    print(f"{'='*80}\n")

    # Output-Video vorbereiten
    out_writer = None
    if not args.no_save:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Output wird gespeichert: {output_path}\n")

    # Metriken-Tracker initialisieren
    metrics = MetricsTracker()
    paused = False

    # Fenster erstellen
    window_name = "YOLOv8 SAR Detection - Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        print("🚀 Starte Live-Inferenz...")
        print("   Steuerung: SPACE = Pause/Resume | Q = Beenden\n")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO Inferenz mit Zeitmessung
                t_start = time.time()
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    max_det=args.max_det,
                    verbose=False,
                    classes=[0],  # Nur Person-Klasse
                )
                inference_time = time.time() - t_start

                r0 = results[0]
                boxes = r0.boxes

                # Bildfläche für relative Area-Schwelle
                h, w = frame.shape[:2]
                frame_area = float(h * w)
                min_area = args.min_area_ratio * frame_area

                filtered_boxes = []

                if boxes is not None:
                    for box in boxes:
                        # xyxy-Koordinaten holen (Tensor -> float)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_w = x2 - x1
                        box_h = y2 - y1
                        area = box_w * box_h

                        if area >= min_area:
                            filtered_boxes.append(box)

                # Anzahl Detektionen nach Area-Filter
                num_detections = len(filtered_boxes)

                # Manuell zeichnen (statt r0.plot())
                annotated_frame = frame.copy()
                for box in filtered_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = float(box.conf[0])
                    label = f"{conf:.2f}"

                    # Bounding Box zeichnen
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    # Label
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Metriken aktualisieren
                metrics.update(num_detections, inference_time)

                # Metriken-Panel zeichnen
                annotated_frame = draw_metrics_panel(
                    annotated_frame,
                    metrics,
                    total_frames,
                    num_detections,
                    args.conf,
                    args.min_area_ratio,
                )

                # Speichern
                if out_writer:
                    out_writer.write(annotated_frame)

                # Live-Anzeige
                cv2.imshow(window_name, annotated_frame)

            # Tastatur-Eingabe verarbeiten
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q oder ESC
                print("\n⚠️  Abbruch durch Benutzer")
                break
            elif key == ord(" "):  # SPACE
                paused = not paused
                status = "PAUSIERT" if paused else "FORTGESETZT"
                print(f"   {status}")

    except KeyboardInterrupt:
        print("\n⚠️  Abbruch durch Benutzer (Ctrl+C)")

    finally:
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

        # Zusammenfassung
        print(f"\n{'='*80}")
        print("✅ ZUSAMMENFASSUNG")
        print(f"{'='*80}")
        print(f"   Verarbeitete Frames:     {metrics.frame_count} / {total_frames}")
        print(f"   Gesamt-Detektionen:      {metrics.total_detections}")
        print(
            f"   Durchschnitt/Frame:      {metrics.get_avg_detections():.2f} Personen"
        )
        print(f"   Maximum/Frame:           {metrics.get_max_detections()} Personen")
        print(f"   Durchschnittliche FPS:   {metrics.get_fps():.1f}")
        print(
            f"   Durchschn. Inferenz:     {metrics.get_avg_inference_time()*1000:.1f}ms"
        )
        print(
            f"   Gesamtlaufzeit:          {int(metrics.get_elapsed_time()//60)}:"
            f"{int(metrics.get_elapsed_time()%60):02d} min"
        )
        if not args.no_save and out_writer:
            print(f"   Output gespeichert:      {output_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
