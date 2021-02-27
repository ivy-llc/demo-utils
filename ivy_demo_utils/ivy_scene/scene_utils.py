# global
import os
import ivy
import math
import ivy_mech
import ivy_vision
import numpy as np
from ivy_vision.containers import PrimitiveScene

# pyrep
try:
    from pyrep import PyRep
    from pyrep.backend import utils
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.arm import Arm
    from pyrep.const import PrimitiveShape
    from pyrep.objects.camera import Camera
    from pyrep.objects.vision_sensor import VisionSensor
    from pyrep.objects.cartesian_path import CartesianPath
    from pyrep.sensors.spherical_vision_sensor import SphericalVisionSensor
except ImportError:
    print('\nPyRep appears to not be installed. For demos with an interactive simulator, please install PyRep.\n')
    PyRep, Dummy, Shape, Arm, PrimitiveShape, Camera, VisionSensor, CartesianPath, SphericalVisionSensor =\
        tuple([None]*9)


class SimObj:

    def __init__(self, pr_obj, f):
        self._pr_obj = pr_obj
        self._f = f

    def get_pos(self):
        return self._f.array(self._pr_obj.get_position(), 'float32')

    def set_pos(self, pos):
        return self._pr_obj.set_position(self._f.to_numpy(pos))

    def set_rot_mat(self, rot_mat):
        inv_ext_mat = np.concatenate((self._f.to_numpy(rot_mat),
                                      np.reshape(self._pr_obj.get_position(), (3, 1))), -1)
        self._pr_obj.set_matrix(inv_ext_mat.reshape((-1,)).tolist())

    def get_inv_ext_mat(self):
        return self._f.reshape(self._f.array(self._pr_obj.get_matrix(), 'float32'), (3, 4))

    def get_ext_mat(self):
        return self._f.inv(ivy_mech.make_transformation_homogeneous(self.get_inv_ext_mat()))[0:3, :]


class SimCam(SimObj):

    def __init__(self, pr_obj, f):
        super().__init__(pr_obj, f)
        self._img_dims = pr_obj.get_resolution()
        if isinstance(pr_obj, VisionSensor):
            pp_offsets = self._f.array([item/2 - 0.5 for item in self._img_dims], 'float32')
            persp_angles = self._f.array([pr_obj.get_perspective_angle() * math.pi/180]*2, 'float32')
            intrinsics = ivy_vision.persp_angles_and_pp_offsets_to_intrinsics_object(
                persp_angles, pp_offsets, self._img_dims)
            self.calib_mat = intrinsics.calib_mats
            self.inv_calib_mat = intrinsics.inv_calib_mats

    def cap(self):
        self._pr_obj.handle_explicitly()
        return self._f.expand_dims(self._f.array(self._pr_obj.capture_depth(True).tolist()), -1),\
               self._f.array(self._pr_obj.capture_rgb().tolist())


# noinspection PyProtectedMember
class BaseSimulator:

    def __init__(self, interactive, try_use_sim, f):
        self._interactive = interactive
        self._try_use_sim = try_use_sim
        self._f = f
        if PyRep is not None and try_use_sim:
            self.with_pyrep = True
            self._pyrep_init()
        else:
            self.with_pyrep = False

    def _user_prompt(self, str_in):
        if self._interactive:
            input(str_in)
        else:
            print(str_in)

    def _pyrep_init(self):

        # pyrep
        self._pyrep = PyRep()
        scene_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scene.ttt')
        self._pyrep.launch(scene_filepath, headless=False, responsive_ui=True)

        # target
        self._target = Dummy('target')

        # default camera
        self._default_camera = Camera('DefaultCamera')

        # default vision sensor
        self._default_vision_sensor = VisionSensor('DefaultVisionSensor')

        # vision sensors
        self._vision_sensor_0 = VisionSensor('vision_sensor_0')
        self._vision_sensor_1 = VisionSensor('vision_sensor_1')
        self._vision_sensor_2 = VisionSensor('vision_sensor_2')
        self._vision_sensor_3 = VisionSensor('vision_sensor_3')
        self._vision_sensor_4 = VisionSensor('vision_sensor_4')
        self._vision_sensor_5 = VisionSensor('vision_sensor_5')
        self._vision_sensors = [self._vision_sensor_0, self._vision_sensor_1, self._vision_sensor_2,
                                self._vision_sensor_3, self._vision_sensor_4, self._vision_sensor_5]

        # vision sensor bodies
        self._vision_sensor_body_0 = Shape('vision_sensor_body_0')
        self._vision_sensor_body_1 = Shape('vision_sensor_body_1')
        self._vision_sensor_body_2 = Shape('vision_sensor_body_2')
        self._vision_sensor_body_3 = Shape('vision_sensor_body_3')
        self._vision_sensor_body_4 = Shape('vision_sensor_body_4')
        self._vision_sensor_body_5 = Shape('vision_sensor_body_5')
        self._vision_sensor_bodies =\
            [self._vision_sensor_body_0, self._vision_sensor_body_1, self._vision_sensor_body_2,
             self._vision_sensor_body_3, self._vision_sensor_body_4, self._vision_sensor_body_5]

        # vision sensor rays
        self._vision_sensor_rays_0 = [Shape('ray{}vs0'.format(i)) for i in range(4)]
        self._vision_sensor_rays_1 = [Shape('ray{}vs1'.format(i)) for i in range(4)]
        self._vision_sensor_rays_2 = [Shape('ray{}vs2'.format(i)) for i in range(4)]
        self._vision_sensor_rays_3 = [Shape('ray{}vs3'.format(i)) for i in range(4)]
        self._vision_sensor_rays_4 = [Shape('ray{}vs4'.format(i)) for i in range(4)]
        self._vision_sensor_rays_5 = [Shape('ray{}vs5'.format(i)) for i in range(4)]
        self._vision_sensor_rays = [self._vision_sensor_rays_0, self._vision_sensor_rays_1, self._vision_sensor_rays_2,
                                    self._vision_sensor_rays_3, self._vision_sensor_rays_4, self._vision_sensor_rays_5]

        # objects
        self._dining_chair_0 = Shape('diningChair0')
        self._dining_chair_1 = Shape('diningChair1')
        self._dining_table = Shape('diningTable_visible')
        self._high_table_0 = Shape('highTable0')
        self._high_table_1 = Shape('highTable1')
        self._plant = Shape('indoorPlant_visible')
        self._sofa = Shape('sofa')
        self._swivel_chair = Shape('swivelChair')
        self._rack = Shape('_rack')
        self._cupboard = Shape('cupboard')
        self._box = Shape('Cuboid')
        self._objects = [self._dining_chair_0, self._dining_chair_1, self._dining_table, self._high_table_0,
                         self._high_table_1, self._plant, self._sofa, self._swivel_chair, self._rack, self._cupboard,
                         self._box]

        # spherical vision sensor
        self._spherical_vision_sensor = SphericalVisionSensor('sphericalVisionRGBAndDepth')

        # drone
        self._drone = Shape('Quadricopter')

        # robot
        self._robot = Arm(0, 'Mico', 6)
        self._robot_base = Dummy('Mico_dh_base')
        self._robot_target = Arm(0, 'MicoTarget', 6)

        # spline paths
        self._spline_paths = list()

        # primitive scene
        self._with_primitive_scene_vis = False

    def _update_path_visualization_pyrep(self, multi_spline_points, multi_spline_sdf_vals):
        if len(self._spline_paths) > 0:
            for spline_path_segs in self._spline_paths:
                for spline_path_seg in spline_path_segs:
                    spline_path_seg.remove()
            self._spline_paths.clear()
        for spline_points, sdf_vals in zip(multi_spline_points, multi_spline_sdf_vals):
            sdf_flags_0 = sdf_vals[1:, 0] > 0
            sdf_flags_1 = sdf_vals[:-1, 0] > 0
            sdf_borders = sdf_flags_1 != sdf_flags_0
            borders_indices = self._f.indices_where(sdf_borders)
            if borders_indices.shape[0] != 0:
                to_concat = (self._f.array([0], 'int32'), self._f.cast(borders_indices, 'int32')[:, 0],
                             self._f.array([-1], 'int32'))
            else:
                to_concat = (self._f.array([0], 'int32'), self._f.array([-1], 'int32'))
            border_indices = self._f.concatenate(to_concat, 0)
            num_groups = border_indices.shape[0] - 1
            spline_path = list()
            for i in range(num_groups):
                border_idx_i = int(self._f.to_numpy(border_indices[i]).item())
                border_idx_ip1 = int(self._f.to_numpy(border_indices[i + 1]).item())
                if i < num_groups - 1:
                    control_group = spline_points[border_idx_i:border_idx_ip1]
                    sdf_group = sdf_vals[border_idx_i:border_idx_ip1]
                else:
                    control_group = spline_points[border_idx_i:]
                    sdf_group = sdf_vals[border_idx_i:]
                num_points = control_group.shape[0]
                orientation_zeros = np.zeros((num_points, 3))
                color = (0.2, 0.8, 0.2) if sdf_group[-1] > 0 else (0.8, 0.2, 0.2)
                control_poses = np.concatenate((ivy.to_numpy(control_group), orientation_zeros), -1)
                spline_path_seg = CartesianPath.create(show_orientation=False, show_position=False, line_size=8,
                                                       path_color=color)
                spline_path_seg.insert_control_points(control_poses.tolist())
                spline_path.append(spline_path_seg)
            self._spline_paths.append(spline_path)

    def depth_to_xyz(self, depth, inv_ext_mat, inv_calib_mat, img_dims):
        uniform_pixel_coords = ivy_vision.create_uniform_pixel_coords_image(img_dims, f=self._f)
        pixel_coords = uniform_pixel_coords * depth
        cam_coords = ivy_vision.pixel_to_cam_coords(pixel_coords, inv_calib_mat, [], img_dims, f=self._f)
        return ivy_vision.cam_to_world_coords(cam_coords, inv_ext_mat)[..., 0:3]

    def get_pix_coords(self):
        return ivy_vision.create_uniform_pixel_coords_image([360, 720], f=self._f)[..., 0:2]

    def setup_primitive_scene_no_sim(self, box_pos=None):

        # lists
        shape_matrices_list = list()
        shape_dims_list = list()

        this_dir = os.path.dirname(os.path.realpath(__file__))
        for i in range(11):
            shape_mat = np.load(os.path.join(this_dir, 'no_sim/obj_inv_ext_mat_{}.npy'.format(i)))
            if i == 10 and box_pos is not None:
                shape_mat[..., -1:] = box_pos.reshape((1, 3, 1))
            shape_matrices_list.append(self._f.array(shape_mat, 'float32'))
            shape_dims_list.append(
                self._f.array(np.load(os.path.join(this_dir, 'no_sim/obj_bbx_{}.npy'.format(i))), 'float32')
            )

        # matices
        shape_matrices = self._f.concatenate(shape_matrices_list, 0)
        shape_dims = self._f.concatenate(shape_dims_list, 0)

        # sdf
        primitive_scene = PrimitiveScene(cuboid_ext_mats=ivy.inv(ivy_mech.make_transformation_homogeneous(
            shape_matrices))[..., 0:3, :], cuboid_dims=shape_dims, f=self._f)
        self.sdf = primitive_scene.sdf

    def setup_primitive_scene(self):

        # shape matrices
        shape_matrices = self._f.concatenate([self._f.reshape(self._f.array(obj.get_matrix(), 'float32'), (1, 3, 4))
                                              for obj in self._objects], 0)

        # shape dims
        x_dims = self._f.concatenate([self._f.reshape(self._f.array(
            obj.get_bounding_box()[1] - obj.get_bounding_box()[0], 'float32'), (1, 1)) for obj in self._objects], 0)
        y_dims = self._f.concatenate([self._f.reshape(self._f.array(
            obj.get_bounding_box()[3] - obj.get_bounding_box()[2], 'float32'), (1, 1)) for obj in self._objects], 0)
        z_dims = self._f.concatenate([self._f.reshape(self._f.array(
            obj.get_bounding_box()[5] - obj.get_bounding_box()[4], 'float32'), (1, 1)) for obj in self._objects], 0)
        shape_dims = self._f.concatenate((x_dims, y_dims, z_dims), -1)

        # primitve scene visualization
        if self._with_primitive_scene_vis:
            scene_vis = [Shape.create(PrimitiveShape.CUBOID, ivy.to_numpy(shape_dim).tolist())
                         for shape_dim in shape_dims]
            [obj.set_matrix(ivy.to_numpy(shape_mat).reshape(-1).tolist())
             for shape_mat, obj in zip(shape_matrices, scene_vis)]
            [obj.set_transparency(0.5) for obj in scene_vis]

        # sdf
        primitive_scene = PrimitiveScene(cuboid_ext_mats=ivy.inv(ivy_mech.make_transformation_homogeneous(
            shape_matrices))[..., 0:3, :], cuboid_dims=shape_dims, f=self._f)
        self.sdf = primitive_scene.sdf

    def update_path_visualization(self, multi_spline_points, multi_spline_sdf_vals, img_path):
        if not self.with_pyrep:
            if not self._interactive:
                return
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            plt.ion()
            plt.imshow(mpimg.imread(img_path))
            plt.show()
            plt.pause(0.1)
            plt.ioff()
            return
        with utils.step_lock:
            self._update_path_visualization_pyrep(multi_spline_points, multi_spline_sdf_vals)

    def close(self):
        if self._interactive:
            input('\nPress enter to end demo.\n')
        print('\nClosing simulator...\n')

    # noinspection PyUnresolvedReferences
    def __del__(self):
        if self.with_pyrep:
            self._pyrep.stop()
            self._pyrep.shutdown()
        print('\nDemo finished.\n')
