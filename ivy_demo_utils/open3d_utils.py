# global
import ivy
import threading
import numpy as np
import open3d as o3d


# noinspection PyCallByClass
class Visualizer:

    def __init__(self, cam_ext_mat=None):

        # visualizer
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window()

        # visualizer control
        self._ctr = self._vis.get_view_control()
        self._cam_ext_mat = cam_ext_mat
        self._first_pass = True
        self._cam_pose_initialized = True if cam_ext_mat is None else False

    # Private #

    def _wait_for_enter(self):
        input('press enter to continue...')
        self._pressend_enter = True

    def _listen_for_enter_in_thread(self):
        self._pressend_enter = False
        self._thread = threading.Thread(target=self._wait_for_enter)
        self._thread.start()

    def _join_enter_listener_thread(self):
        self._thread.join()

    # Public #

    def show_point_cloud(self, xyz_data, rgb_data, interactive, sphere_inv_ext_mats=None, sphere_radii=None):
        if not interactive:
            return

        vectors = o3d.utility.Vector3dVector(np.reshape(ivy.to_numpy(xyz_data), (-1, 3)))
        color_vectors = o3d.utility.Vector3dVector(np.reshape(ivy.to_numpy(rgb_data), (-1, 3)))

        sphere_inv_ext_mats = list() if sphere_inv_ext_mats is None else sphere_inv_ext_mats
        sphere_radii = list() if sphere_radii is None else sphere_radii

        if self._first_pass:
            # create point cloud
            self._point_cloud = o3d.geometry.PointCloud(vectors)
            self._point_cloud.colors = color_vectors
            self._vis.clear_geometries()
            self._vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.15, [0., 0., 0.]), True)
            self._vis.add_geometry(self._point_cloud, True)
            # spheres
            self._spheres = list()
            for sphere_inv_ext_mat, sphere_rad in zip(sphere_inv_ext_mats, sphere_radii):
                sphere = o3d.geometry.TriangleMesh.create_sphere(sphere_rad)
                sphere.paint_uniform_color(np.array([[0.], [0.], [0.]]))
                sphere.transform(sphere_inv_ext_mat)
                self._spheres.append(sphere)
                self._vis.add_geometry(sphere, True)
        else:
            # update point cloud
            self._point_cloud.points = vectors
            self._point_cloud.colors = color_vectors
            self._vis.update_geometry(self._point_cloud)
            for sphere, sphere_inv_ext_mat in zip(self._spheres, sphere_inv_ext_mats):
                sphere.transform(sphere_inv_ext_mat)
                self._vis.update_geometry(sphere)

        # camera matrix
        if not self._cam_pose_initialized:
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.extrinsic = self._cam_ext_mat
            cam_params.intrinsic = self._ctr.convert_to_pinhole_camera_parameters().intrinsic
            self._ctr.convert_from_pinhole_camera_parameters(cam_params)
            self._cam_pose_initialized = True

        # update flag
        self._first_pass = False

        # spin visualizer until key-pressed
        self._listen_for_enter_in_thread()
        while not self._pressend_enter:
            self._vis.poll_events()
        self._join_enter_listener_thread()

        # reset spheres to origin
        for sphere, sphere_inv_ext_mat in zip(self._spheres, sphere_inv_ext_mats):
            sphere.transform(np.linalg.inv(sphere_inv_ext_mat))

    # noinspection PyArgumentList
    def show_voxel_grid(self, voxels, interactive, cuboid_inv_ext_mats=None, cuboid_dims=None):

        if not interactive:
            return

        cuboid_inv_ext_mats = list() if cuboid_inv_ext_mats is None else cuboid_inv_ext_mats
        cuboid_dims = list() if cuboid_dims is None else cuboid_dims

        voxel_grid_data = ivy.to_numpy(voxels[0])
        res = ivy.to_numpy(voxels[2])
        bb_mins = ivy.to_numpy(voxels[3])
        rgb_grid = voxel_grid_data[..., 3:6]
        occupancy_grid = voxel_grid_data[..., -1:]

        boxes = list()
        for x, (x_slice, x_col_slice) in enumerate(zip(occupancy_grid, rgb_grid)):
            for y, (y_slice, y_col_slice) in enumerate(zip(x_slice, x_col_slice)):
                for z, (z_slice, z_col_slice) in enumerate(zip(y_slice, y_col_slice)):
                    if z_slice[0] > 0:
                        box = o3d.geometry.TriangleMesh.create_box(res[0], res[1], res[2])
                        box.vertex_colors = o3d.utility.Vector3dVector(np.ones((8, 3)) * z_col_slice)
                        xtrue = bb_mins[0] + res[0]*x
                        ytrue = bb_mins[1] + res[1]*y
                        ztrue = bb_mins[2] + res[2]*z
                        box.translate(np.array([xtrue, ytrue, ztrue]) - res/2)
                        boxes.append(box)

        all_vertices = np.concatenate([np.asarray(box.vertices) for box in boxes], 0)
        all_vertex_colors = np.concatenate([np.asarray(box.vertex_colors) for box in boxes], 0)
        all_triangles = np.concatenate([np.asarray(box.triangles) + i*8 for i, box in enumerate(boxes)], 0)
        final_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(all_vertices),
                                               o3d.utility.Vector3iVector(all_triangles))
        final_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)

        # add to visualizer
        self._vis.clear_geometries()
        self._vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.15, [0., 0., 0.]), self._first_pass)
        self._vis.add_geometry(final_mesh, self._first_pass)

        # cuboids
        self._cuboids = list()
        for cuboid_inv_ext_mat, cuboid_dim in zip(cuboid_inv_ext_mats, cuboid_dims):
            cuboid = o3d.geometry.TriangleMesh.create_box(cuboid_dim[0], cuboid_dim[1], cuboid_dim[2])
            cuboid.translate(-cuboid_dim/2)
            cuboid.paint_uniform_color(np.array([[0.], [0.], [0.]]))
            cuboid.transform(cuboid_inv_ext_mat)
            self._cuboids.append(cuboid)
            self._vis.add_geometry(cuboid, self._first_pass)

        # camera matrix
        if not self._cam_pose_initialized:
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.extrinsic = self._cam_ext_mat
            cam_params.intrinsic = self._ctr.convert_to_pinhole_camera_parameters().intrinsic
            self._ctr.convert_from_pinhole_camera_parameters(cam_params)
            self._cam_pose_initialized = True

        # update flag
        self._first_pass = False

        # spin visualizer until key-pressed
        self._listen_for_enter_in_thread()
        while not self._pressend_enter:
            self._vis.poll_events()
        self._join_enter_listener_thread()
