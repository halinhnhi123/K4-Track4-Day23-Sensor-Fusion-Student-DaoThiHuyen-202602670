# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đào Thị Huyền
- MSSV: 202602670
- Email: 26ai.huyendt2@vinuni.edu.vn
- Link repo (fork): https://github.com/halinhnhi123/K4-Track4-Day23-Sensor-Fusion-Student-DaoThiHuyen-202602670.git
- Commit hash nộp (`git rev-parse HEAD`): Lấy sau commit cuối bằng `git rev-parse HEAD` và nộp cùng URL fork trên VLearn.

## Tóm tắt kết quả

- `fusion_mode=compare`, `frames=[0, 198]`, `segment=training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed=0`.
- Detection: precision `0.9701`, recall `0.7004`, TP/FP/FN=`519/16/222`.
- LiDAR: RMSE `0.1503 m`, matches `502`, sum_sq_err `11.3436`, ghost_track_frames `0`, missed_gt_frames `239`, mean_confirmed_tracks `2.5226`.
- Fused: RMSE `0.1359 m`, matches `502`, sum_sq_err `9.2668`, ghost_track_frames `0`, missed_gt_frames `239`, mean_confirmed_tracks `2.5226`.
- Rubric cross-check: precision_track=`1.000`, coverage=`502/519=0.9672` for both modes; q=`1.000` for both. `RMSE_fused - RMSE_lidar = -0.0145 m`, within the allowed `0.05 m`.
- Diễn giải: Hai mode có cùng matches, ghosts và misses; fused có RMSE thấp hơn khoảng `0.0145 m`. RMSE thấp một mình không mô tả đầy đủ coverage; cả hai mode đều bỏ lỡ `239` ground-truth frames. Camera trong lab dùng tâm hộp FRONT từ nhãn ground truth có nhiễu seeded, không phải camera detector độc lập.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H)

1. **Đo và covariance:** LiDAR cung cấp `z=[x,y,z]` 3D; `R` là covariance vị trí được đổi từ sensor frame sang vehicle frame khi khởi tạo track. Camera cung cấp `z=[u,v]` theo pixel; EKF dự đoán bằng phép chiếu pinhole và dùng `R=diag(sigma_cam_i², sigma_cam_j²)` theo pixel. Jacobian `H` ánh xạ state sang từng modality.
2. **Mahalanobis gating:** Khoảng cách dùng innovation và covariance `S` để xét residual tương đối theo độ bất định, rồi so với ngưỡng chi-square theo số chiều phép đo. Gating loại các ghép không hợp lý trước update; kiểm tra FOV trước đó cũng ngăn chiếu camera ở tọa độ không hợp lệ. Code: `student/workspace/association.py`.
3. **Track-then-fuse:** Runner dự đoán tracks một lần mỗi frame, sau đó gọi association/update lidar rồi camera trên cùng manager khi có FRONT observations. Điều này thể hiện trong `platform/fusion_lab/scripts/run_lab.py` và log JSONL có các record theo frame/mode; camera không tạo một bộ tracks riêng.
4. **Calibration drift:** Lệch extrinsic/intrinsic tạo innovation camera có bias hoặc residual tăng có hệ thống theo vị trí/góc nhìn; Mahalanobis distance có thể tăng và nhiều phép đo bị gate, hoặc update kéo state sai hướng. Có thể phát hiện bằng cách kiểm tra residual theo frame/vị trí và so với covariance `S`.
5. **Sensor tường minh ở frame rỗng:** Không thể suy ra modality từ `meas_list` khi list rỗng. `sensor` cho biết lượt đó là lidar hay camera để manager vẫn xử lý lidar miss trong FOV, lifecycle/deletion và birth từ unmatched lidar; camera chỉ refine EKF state, không score, spawn hoặc delete tracks. Code: `association.py` và `platform/fusion_lab/tracking/manager.py`.
6. **Vòng đời:** Track mới bắt đầu `initialized` với score `1/window`; hit lidar cộng `1/window` tối đa 1, và score vượt `confirmed_threshold` thì xác nhận. Miss lidar trong FOV trừ `1/window`; trạng thái `confirmed` được giữ qua miss. Xóa nếu `P[0,0]` hoặc `P[1,1] > max_P`, hoặc confirmed có score `< delete_threshold`, hoặc chưa confirmed có score `<= 0`. Code: `student/workspace/track_management.py`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Copilot (AI assistant trong VS Code).
- Dùng cho phần nào: Hỗ trợ hoàn thiện Part E–H, debug lỗi NumPy/native abort và lỗi UTF-8 trên Windows; hỗ trợ tổng hợp báo cáo.
- Cách kiểm tra lại: Chạy `pytest student/tests -q` (128 passed); chạy compare Waymo frame 0–198 với seed 0; xác thực 398 JSONL records bằng `validate_metrics_records`, đối chiếu sáu artifacts và công thức score/coverage theo rubric.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q`: 128 passed, không có failed/ xfailed
- [x] Part A–D: không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đủ sáu artifacts đã được tạo và xác thực; log compare được dựng từ log từng mode đã validate, không chỉnh số liệu
- [x] Điền email và URL fork cá nhân
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP` sau commit cuối
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
