import numpy as np
import cv2
import open3d as o3d

class TrajectoryViewer:

    def __init__(self):

        self.canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        self.origin = (300, 300)

    def update(self, t):

        x = int(t[0][0])
        z = int(t[2][0])

        draw_x = self.origin[0] + x
        draw_y = self.origin[1] - z

        cv2.circle(self.canvas, (draw_x, draw_y), 2, (0, 255, 0), -1)

        cv2.imshow("Trajectory", self.canvas)

class PointCloudViewer:

    def __init__(self, max_points: int = 200_000, point_size: float = 2.0):

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Point Cloud")

        self.pcd = o3d.geometry.PointCloud()
        self._geometry_added = False
        self._initialized_view = False
        self.max_points = int(max_points)
        opt = self.vis.get_render_option()
        if opt is not None:
            opt.point_size = float(point_size)
            opt.background_color = np.asarray([0.0, 0.0, 0.0])

        self._axes_added = False

    def update(self, points):

        if len(points) == 0:
            return

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return

        # Filter invalid points (NaN/Inf)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return

        # Keep visualization responsive
        if self.max_points > 0 and pts.shape[0] > self.max_points:
            pts = pts[-self.max_points :]

        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.paint_uniform_color([0.2, 0.9, 0.2])

        if not self._geometry_added:
            self.vis.add_geometry(self.pcd)
            self._geometry_added = True

        if not self._axes_added:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            self.vis.add_geometry(axes)
            self._axes_added = True

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        if not self._initialized_view:
            self.vis.reset_view_point(True)
            self._initialized_view = True