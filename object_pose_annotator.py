# Author: Seunghyeok Back (shback@kimm.re.kr)
# KIMM, Republic of Korea
# Base codes from Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import shutil
import os
import json
import cv2
import datetime
import numpy as np
import os
import sys
import copy
import matplotlib
import matplotlib.cm

from pathlib import Path
from os.path import basename, dirname
from scipy.spatial.transform import Rotation as Rot

# customize the threshold factor for each camera if needed
camera_idx_to_thresh_factor = {
    0: 1.0, # zivid
    1: 1.0, # realsense d415
    2: 1.0, # realsense d435
    3: 1.0, # azure kinect
}

class Dataset:
    def __init__(self, dataset_path, dataset_split):
        self.scenes_path = os.path.join(dataset_path, dataset_split)
        self.objects_path = os.path.join(dataset_path, 'models_eval')
        self.mesh_path = os.path.join(dataset_path, 'models_obj_eval')


class AnnotationScene:
    def __init__(self, scene_point_cloud, scene_num, image_num):
        self.annotation_scene = scene_point_cloud
        self.scene_num = scene_num
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_mesh, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_mesh, obj_name, obj_instance, transform))

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

    class SceneObject:
        def __init__(self, obj_geometry, obj_mesh, obj_name, obj_instance, transform):
            self.obj_geometry = obj_geometry
            self.obj_mesh = obj_mesh
            self.obj_name = obj_name
            self.obj_instance = obj_instance
            self.transform = np.identity(4)
            self.set_transform(transform)

        def set_transform(self, transform):
            self.obj_geometry.transform(transform)
            self.obj_mesh.transform(transform)
            self.transform = np.matmul(transform, self.transform)


class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.show_coord_frame = False
        self.show_mesh_names = False
        self.highlight_obj = True
        self.transparency = 0.5

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1 - self.transparency]
        self.annotation_obj_material.shader = Settings.UNLIT

        self.annotation_active_obj_material = rendering.MaterialRecord()
        self.annotation_active_obj_material.base_color = [0.3, 0.9, 0.3, 1 - self.transparency]
        self.annotation_active_obj_material.shader = Settings.UNLIT
        
        self.coord_material = rendering.MaterialRecord()
        self.coord_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.coord_material.shader = Settings.UNLIT


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    MATERIAL_SHADERS = [
        Settings.UNLIT
    ]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            if self._scene.scene.has_geometry("annotation_scene"):
                self._scene.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)
                self.settings.apply_material = False
        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._show_coord_frame.checked = self.settings.show_coord_frame
        self._show_mesh_names.checked = self.settings.show_mesh_names
        self._point_size.double_value = self.settings.scene_material.point_size

        if self.settings.show_coord_frame:
            self._add_coord_frame("obj_coord_frame", size=0.1)
            self._add_coord_frame("world_coord_frame")
        else:
            self._scene.scene.remove_geometry("obj_coord_frame")
            self._scene.scene.remove_geometry("world_coord_frame")
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []

        if self.settings.show_mesh_names:
            try:
                self._update_and_show_mesh_name()
            except:
                self._on_error("Scene not loaded yet. (error at _apply_settings)")
                pass
        else:
            for inst_label in self.mesh_names:
                self._scene.remove_3d_label(inst_label)

    def _update_and_show_mesh_name(self):
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        for inst_label in self.mesh_names:
            self._scene.remove_3d_label(inst_label)
        for mesh in meshes:
            self.mesh_names.append(self._scene.add_3d_label(mesh.transform[:3, 3], mesh.obj_name))

    def _add_coord_frame(self, name="coord_frame", size=0.2, origin=[0, 0, 0]):
        if self._annotation_scene is None: # shsh
            self._on_error("Scene not loaded yet. (error at _add_coord_frame)")
            return
        objects = self._annotation_scene.get_objects()
        try:
            active_obj = objects[self._meshes_used.selected_index]
        except IndexError:
            self._on_error("Select an object first. (error at _add_coord_frame)")
            return
        self._scene.scene.remove_geometry(name)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        if "world" in name:
            pass 
            transform = np.eye(4)
            transform[:3, 3] = active_obj.transform[:3, 3]
            coord_frame.transform(transform)
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []
            size = size * 0.6
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([size, 0, 0]), "D (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([-size, 0, 0]), "A (-)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, size, 0]), "S (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, -size, 0]), "W (-)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, 0, size]), "Q (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, 0, -size]), "E (-)"))

        else:
            coord_frame.transform(active_obj.transform)
        self._scene.scene.add_geometry(name, coord_frame, 
                                        self.settings.coord_material,
                                        add_downsampled_copy_for_fast_rendering=True) 

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width_set = 17 * layout_context.theme.font_size
        height_set = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width_set, r.y, width_set,
                                              height_set)

        width_val = 20 * layout_context.theme.font_size
        height_val = min(
            r.height,
            self._validation_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
    
        self._validation_panel.frame = gui.Rect(r.get_right() - width_set - width_val, r.y, width_val,
                                              height_val)
                                        
        width_im = min(
            r.width,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).width * 1.1)
        height_im = min(
            r.height,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height * 1.1)                     
        self._images_panel.frame = gui.Rect(0, r.y, width_im, height_im)   
        self.image_panel_xywh = [0, r.y, width_im, height_im]
        width_obj = 1.5 * width_set
        height_obj = 1.5 * layout_context.theme.font_size
        self._log_panel.frame = gui.Rect(0, r.get_bottom() - height_obj, width_obj, height_obj) 

    def __init__(self, width, height):

        self._annotation_scene = None
        self._annotation_changed = False
        self.current_scene_idx = None
        self.current_image_idx = None
        self.upscale_responsiveness = False
        self.scene_obj_info = None
        self.bounds = None
        self.coord_labels = []
        self.mesh_names = []
        self.settings = Settings()
        self.ok_delta = 25
        self.scale_factor = None


        self.window = gui.Application.instance.create_window(
            "6D Object Pose Annotator", width, height)
        w = self.window  

        self.spl = "\\" if sys.platform.startswith("win") else "/"

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        em = w.theme.font_size

        # ---- Validation panel ----
        self._validation_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.scene_obj_info_panel = gui.CollapsableVert("Annotation Quality", 0,
                                         gui.Margins(em, 0, 0, 0))
        self.scene_obj_info_panel.set_is_open(True)
        self.scene_obj_info_table = gui.ListView()
        self.scene_obj_info_panel.add_child(self.scene_obj_info_table)
        self._validation_panel.add_child(self.scene_obj_info_panel)

        self.anno_copy_panel = gui.Vert(
            em, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        source_grid = gui.VGrid(3, 0.25 * em)
        self.source_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.source_id_edit.int_value = 0
        self.source_id_edit.set_limits(-4, 72)
        self.source_id_edit.set_on_value_changed(self._on_source_id_edit)
        self.source_image_num = 0
        source_grid.add_child(gui.Label("Source Image ID:"))
        source_grid.add_child(self.source_id_edit)

        target_grid = gui.VGrid(3, 0.25*em)
        self.target_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.target_id_edit.int_value = 1
        self.target_id_edit.set_limits(-4, 72)
        self.target_id_edit.set_on_value_changed(self._on_target_id_edit)
        self.target_image_num = 1
        target_grid.add_child(gui.Label("Target Image ID:"))
        target_grid.add_child(self.target_id_edit)
        
        self._copy_button  = gui.Button('Copy Annotation')
        self._copy_button.set_on_clicked(self._on_copy_button)
        self.anno_copy_panel.add_child(source_grid)
        self.anno_copy_panel.add_child(target_grid)
        self.anno_copy_panel.add_child(self._copy_button)
        self.scene_obj_info_panel.add_child(self.anno_copy_panel)

        # ---- Settings panel ----
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("Open File")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("File Path:"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        self._settings_panel.add_child(fileedit_layout)

        view_ctrls = gui.CollapsableVert("View Controls", 0.33 * em,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("Show Camera Coordinate Frame")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._highlight_obj = gui.Checkbox("Highlight Active Object")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._show_coord_frame = gui.Checkbox("Show Object Coordinate Frame")
        self._show_coord_frame.set_on_checked(self._on_show_coord_frame)
        view_ctrls.add_child(self._show_coord_frame)

        self._show_mesh_names = gui.Checkbox("Show Object Names")
        self._show_mesh_names.set_on_checked(self._on_show_mesh_names)
        view_ctrls.add_child(self._show_mesh_names)

        self._transparency = gui.Slider(gui.Slider.DOUBLE)
        self._transparency.set_limits(0, 1)
        self._transparency.set_on_value_changed(self._on_transparency)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        self.dist = 0.0004 * 5
        self.deg = 0.2 * 5
        self._responsiveness = gui.Slider(gui.Slider.INT)
        self._responsiveness.set_limits(1, 20)
        self._responsiveness.set_on_value_changed(self._on_responsiveness)
        self._responsiveness.double_value = 5.0

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Transparency"))
        grid.add_child(self._transparency)
        grid.add_child(gui.Label("Point Size"))
        grid.add_child(self._point_size)
        grid.add_child(gui.Label("Responsiveness"))
        grid.add_child(self._responsiveness)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # ----
        self._images_panel = gui.CollapsableVert("Images", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        self._vis_img_proxy = gui.WidgetProxy()
        self._vis_img_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._vis_img_proxy)
        self._images_panel.set_is_open(False)


        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t Start by opening a file.")
        self._log_panel.add_child(self._log)
        self.window.set_needs_layout()

        # 3D Annotation tool options
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._images_panel)
        w.add_child(self._log_panel)
        w.add_child(self._validation_panel)
        w.set_on_layout(self._on_layout)

        annotation_objects = gui.CollapsableVert("Annotation Objects", 0.25 * em,
                                                 gui.Margins(0.25*em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        object_select_layout = gui.VGrid(2)
        object_select_layout.add_child(gui.Label("Object ID: obj_"))
        self._meshes_available = gui.NumberEdit(gui.NumberEdit.INT)
        self._meshes_available.int_value = 1
        self._meshes_available.set_limits(1, 200)
        object_select_layout.add_child(self._meshes_available)
        annotation_objects.add_child(object_select_layout)

        inst_grid = gui.VGrid(3)
        self.inst_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.inst_id_edit.int_value = 1
        self.inst_id_edit.set_limits(1, 30)
        self.inst_id_edit.set_on_value_changed(self._on_inst_value_changed)
        self._meshes_used = gui.ListView()
        self._meshes_used.set_on_selection_changed(self._on_selection_changed)
        add_mesh_button = gui.Button("Add Object")
        add_mesh_button.horizontal_padding_em = 0.8
        add_mesh_button.vertical_padding_em = 0.2
        remove_mesh_button = gui.Button("Remove Object")
        remove_mesh_button.horizontal_padding_em = 0.8
        remove_mesh_button.vertical_padding_em = 0.2
        add_mesh_button.set_on_clicked(self._add_mesh)
        remove_mesh_button.set_on_clicked(self._remove_mesh)

        inst_grid.add_child(gui.Label("Instance ID:"))
        inst_grid.add_child(self.inst_id_edit)
        annotation_objects.add_child(inst_grid)
        hz = gui.Horiz(spacing=5)
        hz.add_child(add_mesh_button)
        hz.add_child(remove_mesh_button)
        annotation_objects.add_child(hz)

        annotation_objects.add_child(self._meshes_used)

        # x, y, z axis
        x_grid = gui.VGrid(3, 0.25 * em)
        self._x_rot = gui.Slider(gui.Slider.DOUBLE)
        self._x_rot.set_limits(-0.5, 0.5)
        self._x_rot.set_on_value_changed(self._on_x_rot)
        x_grid.add_child(gui.Label("X axis",))
        x_grid.add_child(self._x_rot)
        annotation_objects.add_child(x_grid)

        y_grid = gui.VGrid(3, 0.25 * em)
        self._y_rot = gui.Slider(gui.Slider.DOUBLE)
        self._y_rot.set_limits(-0.5, 0.5)
        self._y_rot.set_on_value_changed(self._on_y_rot)
        y_grid.add_child(gui.Label("Y axis",))
        y_grid.add_child(self._y_rot)
        annotation_objects.add_child(y_grid)

        z_grid = gui.VGrid(3, 0.25 * em)
        self._z_rot = gui.Slider(gui.Slider.DOUBLE)
        self._z_rot.set_limits(-0.5, 0.5)
        self._z_rot.set_on_value_changed(self._on_z_rot)
        z_grid.add_child(gui.Label("Z axis",))
        z_grid.add_child(self._z_rot)
        annotation_objects.add_child(z_grid)
        self._settings_panel.add_child(annotation_objects)



        self._scene_control = gui.CollapsableVert("File Control", 0.25 * em,
                                                  gui.Margins(0.25 * em, 0, 0, 0))
        self._scene_control.set_is_open(True)

        h = gui.Horiz(0.4 * em)  
        self.image_number_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.image_number_edit.int_value = 1
        self.image_number_edit.set_limits(-4, 53) #!TODO: update this automatically for BOP dataset
        h.add_child(gui.Label("Image:"))
        h.add_child(self.image_number_edit)
        change_image_button = gui.Button("Change")
        change_image_button.set_on_clicked(self._on_change_image)
        change_image_button.horizontal_padding_em = 0.8
        change_image_button.vertical_padding_em = 0
        h.add_child(change_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        self._images_buttons_label = gui.Label("Image:")
        self._samples_buttons_label = gui.Label("Directory:")

        self._pre_image_button = gui.Button("Previous")
        self._pre_image_button.horizontal_padding_em = 0.8
        self._pre_image_button.vertical_padding_em = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)
        self._next_image_button = gui.Button("Next")
        self._next_image_button.horizontal_padding_em = 0.8
        self._next_image_button.vertical_padding_em = 0
        self._next_image_button.set_on_clicked(self._on_next_image)
        self._pre_sample_button = gui.Button("Previous")
        self._pre_sample_button.horizontal_padding_em = 0.8
        self._pre_sample_button.vertical_padding_em = 0
        self._pre_sample_button.set_on_clicked(self._on_previous_scene)
        self._next_sample_button = gui.Button("Next")
        self._next_sample_button.horizontal_padding_em = 0.8
        self._next_sample_button.vertical_padding_em = 0
        self._next_sample_button.set_on_clicked(self._on_next_scene)
        # 2 rows for sample and scene control
        h = gui.Horiz(0.4 * em)  # row 1
        h.add_child(self._images_buttons_label)
        h.add_child(self._pre_image_button)
        h.add_child(self._next_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)
        h = gui.Horiz(0.4 * em)  # row 2
        h.add_child(self._samples_buttons_label)
        h.add_child(self._pre_sample_button)
        h.add_child(self._next_sample_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        self._view_numbers = gui.Horiz(0.4 * em)
        self._image_number = gui.Label("Image: " + f'{0:06}')
        self._scene_number = gui.Label("Directory: " + f'{0:06}')

        self._view_numbers.add_child(self._image_number)
        self._view_numbers.add_child(self._scene_number)
        self._scene_control.add_child(self._view_numbers)


        progress_ctrls = gui.Vert(em)
        self._progress = gui.ProgressBar()
        self._progress.value = 0.0 
        prog_layout = gui.Vert(em)
        prog_layout.add_child(self._progress)
        self._progress_str = gui.Label("Progress: 0.0% [0/0]")
        progress_ctrls.add_child(self._progress_str)
        progress_ctrls.add_child(self._progress)
        self._scene_control.add_child(progress_ctrls)


        self._settings_panel.add_child(self._scene_control)
        initial_viewpoint = gui.Button("Move to Initial Viewpoint (T)")
        initial_viewpoint.horizontal_padding_em = 0.8
        initial_viewpoint.vertical_padding_em = 0.2
        initial_viewpoint.set_on_clicked(self._on_initial_viewpoint)
        self._scene_control.add_child(initial_viewpoint)
        refine_position = gui.Button("Refine Pose using ICP (R)")
        refine_position.horizontal_padding_em = 0.8
        refine_position.vertical_padding_em = 0.2
        refine_position.set_on_clicked(self._on_refine)
        self._scene_control.add_child(refine_position)
        generate_save_annotation = gui.Button("Save Annotation")
        generate_save_annotation.horizontal_padding_em = 0.8
        generate_save_annotation.vertical_padding_em = 0.2
        generate_save_annotation.set_on_clicked(self._on_generate)
        self._scene_control.add_child(generate_save_annotation)
        
        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("About", help_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        
        # ---- annotation tool settings ----
        self._on_transparency(0.5)
        self._on_point_size(5)  # set default size to 1
        self._apply_settings()

        # set callbacks for key control
        self._scene.set_on_key(self._transform)
        self._left_shift_modifier = False
        self._scene.set_on_mouse(self._on_mouse)
        self._log.text = "\t Start by opening a file."
        self.window.set_needs_layout()

    def _on_source_id_edit(self, new_val):
        self.source_image_num = int(new_val)
    
    def _on_target_id_edit(self, new_val):
        self.target_image_num = int(new_val)

    def _on_copy_button(self):

        if self._annotation_changed:
            self._on_error('Try again after saving the current annotation. (error at _on_copy_button)')
            return

        if self.source_image_num < 0:
            source_image_num = f'{self.source_image_num:07}'
        else:
            source_image_num = f'{self.source_image_num:06}'
        if self.target_image_num < 0:
            target_image_num = f'{self.target_image_num:07}'
        else:
            target_image_num = f'{self.target_image_num:06}'
        
        if self.source_image_num == self.target_image_num:
            self._on_error('The source and target image numbers are the same. (error at _on_copy_button)')
            return

        self._log.text = "Copying the labeling result of image " + source_image_num + " to " + target_image_num + "..."
        self.window.set_needs_layout()

        json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", 'scene_gt.json')
        if not os.path.exists(json_6d_path):
            self._on_error('The json file does not exist. (error at _on_copy_button)')
            return
        with open(json_6d_path, "r") as gt_scene:
            try:
                gt_6d_pose_data = json.load(gt_scene)
            except json.decoder.JSONDecodeError as e:
                self._on_error("Error loading the json file. (error at _on_copy_button)")
                return
        
        se3_base_to_source = np.eye(4)
        scene_camera_info = copy.deepcopy(self.scene_camera_info)
        se3_base_to_source[:3, :3] = np.array(scene_camera_info[str(int(self.source_image_num))]["cam_R_w2c"]).reshape(3, 3)
        se3_base_to_source[:3, 3] = np.array(scene_camera_info[str(int(self.source_image_num))]["cam_t_w2c"])


        se3_base_to_target = np.eye(4)
        se3_base_to_target[:3, :3] = np.array(scene_camera_info[str(int(self.target_image_num))]["cam_R_w2c"]).reshape(3, 3)
        se3_base_to_target[:3, 3] = np.array(scene_camera_info[str(int(self.target_image_num))]["cam_t_w2c"])

        se3_target_to_source = np.matmul(np.linalg.inv(se3_base_to_target), se3_base_to_source)

        if str(int(source_image_num)) not in gt_6d_pose_data:
            self._on_error('The source image number does not exist in the json file. (error at _on_copy_button)')
            return

        with open(json_6d_path, 'w+') as gt_scene:
            source_data = copy.deepcopy(gt_6d_pose_data[str(int(source_image_num))])
            target_data = list()
            for source in source_data:
                se3_source_to_object = np.eye(4)
                se3_source_to_object[:3, :3] = copy.deepcopy(np.array(source['cam_R_m2c']).reshape(3, 3))
                se3_source_to_object[:3, 3] = copy.deepcopy(np.array(source['cam_t_m2c']))
                se3_target_to_object = np.matmul(se3_target_to_source, copy.deepcopy(se3_source_to_object))
                target = copy.deepcopy(source)
                target['cam_R_m2c'] = se3_target_to_object[:3, :3].reshape(9).tolist() 
                target['cam_t_m2c'] = se3_target_to_object[:3, 3].tolist()
                target_data.append(target)
            gt_6d_pose_data[str(int(target_image_num))] = target_data
            json.dump(gt_6d_pose_data, gt_scene)
        self._log.text = "\tCopied the annotation of image " + source_image_num + " to " + target_image_num + "."
        self.window.set_needs_layout()


    def update_scene_obj_info_table(self):

        self.scene_obj_info_table_data = []
        target_obj_names = []
        target_obj_inst_names = []
        for obj_inst_name in self.depth_diff_means.keys():
            obj_id = int(obj_inst_name.split('_')[1])
            inst_id = int(obj_inst_name.split('_')[2])
            err = -1
            obj_name = f'obj_{obj_id:06}'
            obj_inst_name = f'obj_{obj_id:06}_{inst_id}'
            target_obj_names.append(obj_name)
            target_obj_inst_names.append(obj_inst_name)
            if obj_inst_name in self.depth_diff_means.keys():
                err = abs(self.depth_diff_means[obj_inst_name]) 
            
            ok_delta = self.ok_delta 
            ok_delta *= camera_idx_to_thresh_factor[self.current_image_idx % 4]
            if err <= ok_delta :
                text = "Complete"
            else:
                text = "Incomplete"
            self.scene_obj_info_table_data.append([f'obj_{obj_id:06}_{inst_id}', text, err])

        scene_obj_info_table = []

        for i, table_data in enumerate(self.scene_obj_info_table_data):
            row = "{}: {} ({:.1f})".format(table_data[0], table_data[1], table_data[2])
            scene_obj_info_table.append(row)
        self.scene_obj_info_table.set_items(scene_obj_info_table)

    def _on_x_rot(self, new_val):
        try:
            self.move( 0, 0, 0, new_val * np.pi / 180, 0, 0)
        except:
            self._on_error("Select an object first. (error at _on_x_rot)")
        self._x_rot.int_value = 0      

    def _on_y_rot(self, new_val):
        try:
            self.move( 0, 0, 0, 0, new_val * np.pi / 180, 0)
        except:
            self._on_error("Select an object first. (error at _on_y_rot)")
        self._y_rot.int_value = 0     

    def _on_z_rot(self, new_val):
        try:
            self.move( 0, 0, 0, 0, 0, new_val * np.pi / 180)
        except:
            self._on_error("Select an object first. (error at _on_z_rot)")
        self._z_rot.int_value = 0     

    def _on_inst_value_changed(self, new_val):
        if int(new_val) < 1:
            self._on_error("Instance ID must be greater than 0. (error at _on_inst_value_changed)")
            return
        idx = self._meshes_used.selected_index
        try:
            obj_name = self._annotation_scene.get_objects()[idx].obj_name
        except AttributeError:
            self._on_error("Select an object first. (error at _on_inst_value_changed)")
            return
        self._annotation_scene.get_objects()[idx].obj_instance = int(new_val)
        self._annotation_scene.get_objects()[idx].obj_name = "obj_" + obj_name.split("_")[1] + "_" + str(int(new_val))
        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = idx
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()
        self._log.text = "\tChanged the instance ID of the object."
        self.window.set_needs_layout()

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Open File",
                                 self.window.theme)
        filedlg.add_filter(".png", "Image (.png)")
        filedlg.add_filter(".jpg", "Image (.jpg)")
        filedlg.add_filter("", "All files (*)")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)


    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        dataset_path = str(Path(path).parent.parent.parent.parent)
        split_and_type = basename(str(Path(path).parent.parent.parent))
        self.scenes = Dataset(dataset_path, split_and_type)

        start_scene_num = int(basename(str(Path(path).parent.parent)))
        start_image_num = int(basename(path)[:-4])
        self.scene_num_lists = sorted([int(basename(x)) for x in glob.glob(dirname(str(Path(path).parent.parent)) + self.spl + "*") if os.path.isdir(x)])
        self.current_scene_idx = self.scene_num_lists.index(start_scene_num)
        self.image_num_lists = sorted([int(basename(x).split(".")[0]) for x in glob.glob(dirname(str(Path(path))) + self.spl + "*.png")])
        if len(self.image_num_lists) == 0:
            self.image_num_lists = sorted([int(basename(x).split(".")[0]) for x in glob.glob(dirname(str(Path(path))) + self.spl + "*.jpg")])
        self.current_image_idx = self.image_num_lists.index(start_image_num)
        if os.path.exists(self.scenes.scenes_path) and os.path.exists(self.scenes.objects_path):
            self.update_obj_list()
            self.scene_load(self.scenes.scenes_path, start_scene_num, start_image_num)
            self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) 
            self._progress_str.text = "Progress: {:.1f}% [{}/{}]".format(
                100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
                self.current_image_idx + 1, len(self.image_num_lists))
        self.window.close_dialog()
        self._log.text = "\tLoad a scene to start annotating."
        self.window.set_needs_layout()

    def _update_scene_numbers(self):
        self._scene_number.text = "Directory: " + f'{self._annotation_scene.scene_num:06}'
        self._image_number.text = "Image: " + f'{self._annotation_scene.image_num:06}'

    def move(self, x, y, z, rx, ry, rz):
        self._annotation_changed = True
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        # translation or rotation
        if x != 0 or y != 0 or z != 0:
            h_transform = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        else: 
            center = active_obj.obj_geometry.get_center()
            rot_mat_obj_center = active_obj.obj_geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
            T_neg = np.vstack((np.hstack((np.identity(3), -center.reshape(3, 1))), [0, 0, 0, 1]))
            R = np.vstack((np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1]))
            T_pos = np.vstack((np.hstack((np.identity(3), center.reshape(3, 1))), [0, 0, 0, 1]))
            h_transform = np.matmul(T_pos, np.matmul(R, T_neg))
            
        active_obj.set_transform(h_transform)
        center = active_obj.obj_geometry.get_center()
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
                                    
        # update values stored of object
        if self.settings.show_coord_frame:
            self._add_coord_frame("obj_coord_frame", size=0.1)
            self._add_coord_frame("world_coord_frame")
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()


    def _transform(self, event):
        if event.key == gui.KeyName.ESCAPE:
            self._on_generate()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
            return gui.Widget.EventCallbackResult.HANDLED

        # if ctrl is pressed then increase translation and angle values
        if event.key == gui.KeyName.LEFT_CONTROL or event.key == gui.KeyName.RIGHT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                if not self.upscale_responsiveness:
                    self.dist = self.dist * 15
                    self.deg = self.deg * 15
                    self.upscale_responsiveness = True
            elif event.type == gui.KeyEvent.UP:
                if self.upscale_responsiveness:
                    self.dist = self.dist / 15
                    self.deg = self.deg / 15
                    self.upscale_responsiveness = False
            return gui.Widget.EventCallbackResult.HANDLED

        if event.key == gui.KeyName.R and event.type == gui.KeyEvent.DOWN:
            self._on_refine()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_initial_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.F and event.type == gui.KeyEvent.DOWN:
            self._on_generate()
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.V and event.type == gui.KeyEvent.DOWN:
            is_open = self._images_panel.get_is_open()
            self._images_panel.set_is_open(not is_open)
            self.window.set_needs_layout()
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.ONE and event.type == gui.KeyEvent.DOWN:
            if self._responsiveness.double_value >= 2:
                self._responsiveness.double_value -= 1
            else:
                self._responsiveness.double_value = 1
            self.dist = 0.0004 * self._responsiveness.double_value
            self.deg = 0.2 * self._responsiveness.double_value
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.TWO and event.type == gui.KeyEvent.DOWN:
            if self._responsiveness.double_value <= 19:
                self._responsiveness.double_value += 1
            else:
                self._responsiveness.double_value = 20
            self.dist = 0.0004 * self._responsiveness.double_value
            self.deg = 0.2 * self._responsiveness.double_value
            return gui.Widget.EventCallbackResult.HANDLED      


        if event.key in [gui.KeyName.I, gui.KeyName.J, gui.KeyName.K, gui.KeyName.L, gui.KeyName.U, gui.KeyName.O, gui.KeyName.P] and self.scale_factor is not None:
            def translate(tx=0, ty=0):
                T = np.eye(3)
                T[0:2,2] = [tx, ty]
                return T
            def scale(s=1, sx=1, sy=1):
                T = np.diag([s*sx, s*sy, 1])
                return T
            def rotate(degrees):
                T = np.eye(3)
                T[0:2] = cv2.getRotationMatrix2D(center=(0,0), angle=-degrees, scale=1.0)
                return T
            translate_factor = 10
            if event.key == gui.KeyName.I:
                self.icy -= translate_factor
            if event.key == gui.KeyName.K:
                self.icy += translate_factor
            if event.key == gui.KeyName.J:
                self.icx -= translate_factor
            if event.key == gui.KeyName.L:
                self.icx += translate_factor
            if event.key == gui.KeyName.U:
                self.scale_factor += 0.1
            if event.key == gui.KeyName.O:
                self.scale_factor -= 0.1
            if event.key == gui.KeyName.P:
                self.icx, self.icy = self.W//2, self.H//2
                self.scale_factor = 1.0
            if self.icy < 0:
                self.icy = 0
            if self.icx < 0:
                self.icx = 0
            if self.icy > self.H:
                self.icy = self.H
            if self.icx > self.W:
                self.icx = self.W
            if self.scale_factor < 0.1:
                self.scale_factor = 0.1
            if self.scale_factor > 10:
                self.scale_factor = 10
            (ow, oh) = (self.W, self.H)
            (ocx, ocy) = ((ow-1)/2, (oh-1)/2)
            H = translate(+ocx, +ocy) @ rotate(degrees=0) @ scale(self.scale_factor) @ translate(-self.icx, -self.icy)
            M = H[0:2]
            
            rgb_img = cv2.resize(self.rgb_img.copy(), self.diff_img.shape[:2][::-1])
            rgb_img = cv2.warpAffine(rgb_img, dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            diff_img = cv2.warpAffine(self.diff_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            mask_img = cv2.warpAffine(self.mask_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            self._update_vis_img(rgb_img, diff_img, mask_img)
            return gui.Widget.EventCallbackResult.HANDLED

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("Select an object first. (error at _transform)")
            return gui.Widget.EventCallbackResult.HANDLED

        # Translation
        if not self._left_shift_modifier:
            self._log.text = "\tAdjusting the object position."
            self.window.set_needs_layout()
            if event.key == gui.KeyName.D:
                self.move( self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.A:
                self.move( -self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.S:
                self.move( 0, self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.W:
                self.move( 0, -self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.Q:
                self.move( 0, 0, self.dist, 0, 0, 0)
            elif event.key == gui.KeyName.E:
                self.move( 0, 0, -self.dist, 0, 0, 0)
        # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
        else:
            self._log.text = "\tAdjusting the object orientation."
            self.window.set_needs_layout()
            if event.key == gui.KeyName.E:
                self.move( 0, 0, 0, 0, 0, self.deg * np.pi / 180)
            elif event.key == gui.KeyName.Q:
                self.move( 0, 0, 0, 0, 0, -self.deg * np.pi / 180)
            elif event.key == gui.KeyName.A:
                self.move( 0, 0, 0, 0, self.deg * np.pi / 180, 0)
            elif event.key == gui.KeyName.D:
                self.move( 0, 0, 0, 0, -self.deg * np.pi / 180, 0)
            elif event.key == gui.KeyName.S:
                self.move( 0, 0, 0, self.deg * np.pi / 180, 0, 0)
            elif event.key == gui.KeyName.W:
                self.move( 0, 0, 0, -self.deg * np.pi / 180, 0, 0)

        return gui.Widget.EventCallbackResult.HANDLED


    def _on_mouse(self, event):
        
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.ALT):
            try:
                objects = self._annotation_scene.get_objects()
                active_obj = objects[self._meshes_used.selected_index]
            except IndexError:
                self._on_error("Select an object first. (error at _on_mouse)")
                return gui.Widget.EventCallbackResult.HANDLED

            def depth_callback(depth_image):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    pass
                else:
                    target_xyz = self._scene.scene.camera.unproject(
                        event.x, event.y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    target_xyz = np.array(target_xyz)
                    objects = self._annotation_scene.get_objects()
                    active_obj = objects[self._meshes_used.selected_index]
                    h_transform = np.eye(4)
                    h_transform[:3, 3] = target_xyz - active_obj.obj_geometry.get_center()
                    active_obj.set_transform(h_transform)
                    self._scene.scene.remove_geometry(active_obj.obj_name)
                    self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                                self.settings.annotation_active_obj_material,
                                                add_downsampled_copy_for_fast_rendering=True)
                    # update values stored of object
                    if self.settings.show_coord_frame:
                        self._add_coord_frame("obj_coord_frame", size=0.1)
                        self._add_coord_frame("world_coord_frame")
                    if self.settings.show_mesh_names:
                        self._update_and_show_mesh_name()
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            self._log.text = "\tAdjusting the object position using mouse click."
            self.window.set_needs_layout()
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.HANDLED

    def _on_selection_changed(self, a, b):
        self._log.text = "\tSelected object: " + str(self._meshes_used.selected_index)
        self.window.set_needs_layout()
        objects = self._annotation_scene.get_objects()
        for obj in objects:
            self._scene.scene.remove_geometry(obj.obj_name)
            self._scene.scene.add_geometry(obj.obj_name, obj.obj_geometry,
                                        self.settings.annotation_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        active_obj = objects[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                    self.settings.annotation_active_obj_material,
                                    add_downsampled_copy_for_fast_rendering=True)
        self.inst_id_edit.set_value(int(active_obj.obj_name.split("_")[-1]))
        self._apply_settings()

    def _on_refine(self):
        self._log.text = "\tRefining the pose using ICP..."
        self.window.set_needs_layout()
        self._annotation_changed = True

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("Select an object first. (error at _on_refine)")
            return gui.Widget.EventCallbackResult.HANDLED

        target = self._annotation_scene.annotation_scene
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        source = active_obj.obj_geometry

        trans_init = np.identity(4)
        threshold = 0.004
        reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=50))
        if np.sum(np.abs(reg.transformation[:3, 3])) < 0.25:
            active_obj.set_transform(reg.transformation)
            self._scene.scene.remove_geometry(active_obj.obj_name)
            self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)

            if self.settings.show_coord_frame:
                self._add_coord_frame("obj_coord_frame", size=0.1)
                self._add_coord_frame("world_coord_frame")
            if self.settings.show_mesh_names:
                self._update_and_show_mesh_name()
            self._log.text = "\tSuccess to refine the pose using ICP."
            self.window.set_needs_layout()
        else:
            self._log.text = "\tFailed to refine the pose. Try again or adjust it manually."
            self.window.set_needs_layout()

    def _on_generate(self):
        self._log.text = "\tSaving the annotation results..."
        self.window.set_needs_layout()
        if self._annotation_scene is None: # shsh
            self._on_error("Select a scene to save the annotation results. (error at _on_generate)")
            return

        image_num = self._annotation_scene.image_num
        json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", 'scene_gt')

        if os.path.exists(json_6d_path):
            with open(json_6d_path, "r") as gt_scene:
                try:
                    gt_6d_pose_data = json.load(gt_scene)
                except json.decoder.JSONDecodeError as e:
                    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = json_6d_path.replace(".json", "_backup_{}.json".format(date_time))
                    shutil.copy(json_6d_path, backup_path)
                    self._on_error("Failed to load the json file. The file is saved as a backup file. (error at _on_generate)")
                    gt_6d_pose_data = {}
        else:
            gt_6d_pose_data = {}

        # write/update "scene_gt.json"
        try:
            with open(json_6d_path, 'w+') as gt_scene:
                view_angle_data = list()
                for obj in self._annotation_scene.get_objects():
                    transform_cam_to_object = obj.transform
                    translation = np.array(transform_cam_to_object[0:3, 3] * 1000, dtype=np.float32).tolist()  # convert meter to mm
                    model_names = self.load_model_names()
                    obj_id = int(obj.obj_name.split("_")[1])  # assuming object name is formatted as obj_000001
                    inst_id = int(obj.obj_name.split("_")[2])
                    obj_data = {
                        "cam_R_m2c": transform_cam_to_object[0:3, 0:3].tolist(),  # rotation matrix
                        "cam_t_m2c": translation,  # translation
                        "obj_id": obj_id,
                        "inst_id": inst_id
                    }
                    view_angle_data.append(obj_data)
                if str(image_num) in gt_6d_pose_data.keys():
                    del gt_6d_pose_data[str(image_num)]
                gt_6d_pose_data[str(image_num)] = view_angle_data
                json.dump(gt_6d_pose_data, gt_scene)
            self._log.text = "\tSave the annotation results successfully."
            self.window.set_needs_layout()
        except Exception as e:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", "scene_gt_backup_{}.json".format(date_time))
            with open(json_6d_path, 'w+') as gt_scene:
                view_angle_data = list()
                for obj in self._annotation_scene.get_objects():
                    transform_cam_to_object = obj.transform
                    translation = list(transform_cam_to_object[0:3, 3] * 1000)  # convert meter to mm
                    model_names = self.load_model_names()
                    obj_id = int(obj.obj_name.split("_")[1])  # assuming object name is formatted as obj_000001
                    inst_id = int(obj.obj_name.split("_")[2])
                    obj_data = {
                        "cam_R_m2c": transform_cam_to_object[0:3, 0:3].tolist(),  # rotation matrix
                        "cam_t_m2c": translation,  # translation
                        "obj_id": obj_id,
                        "inst_id": inst_id
                    }
                    view_angle_data.append(obj_data)
                gt_6d_pose_data[str(image_num)] = view_angle_data
                json.dump(gt_6d_pose_data, gt_scene)
            self._log.text = "\tFailed to save the annotation results. The results are saved as a backup file."
            self.window.set_needs_layout()
        self._annotation_changed = False
        self._validate_anno()
        self.update_scene_obj_info_table()

    def _validate_anno(self):
         # annotation validator
        self._log.text = "\tGenerating validation results..."
        self.window.set_needs_layout()   
        render = rendering.OffscreenRenderer(width=self.W, height=self.H)
        render.scene.set_background([0, 0, 0, 1])
        # adjust cam_K to render size
        intrinsic = np.array(self.cam_K).reshape((3, 3))
        intrinsic[0, 0] *= 1 / 4
        intrinsic[1, 1] *= 1 / 4
        intrinsic[0, 2] *= 1 / 4
        intrinsic[1, 2] *= 1 / 4

        # set camera intrinsic
        extrinsic = np.eye(4)
        render.setup_camera(intrinsic, extrinsic, self.W, self.H)
        # set camera pose
        center = [0, 0, 1]  # look_at target
        eye = [0, 0, 0]  # camera position
        up = [0, -1, 0]  # camera orientation
        render.scene.camera.look_at(center, eye, up)
        render.scene.camera.set_projection(intrinsic, 0.01, 3.0, self.W, self.H)
        
        objects = self._annotation_scene.get_objects()
        # generate object material
        obj_mtl = o3d.visualization.rendering.MaterialRecord()
        obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
        obj_mtl.shader = "defaultUnlit"
        for obj in objects:
            obj = copy.deepcopy(obj)
            obj.obj_mesh.paint_uniform_color([1, 1, 1])
            render.scene.add_geometry(obj.obj_name, obj.obj_mesh, obj_mtl,                              
                                  add_downsampled_copy_for_fast_rendering=False)
        

        depth_rendered = render.render_to_depth_image(z_in_view_space=True)
        depth_rendered = np.array(depth_rendered, dtype=np.float32)
        depth_rendered[np.isposinf(depth_rendered)] = 0
        depth_rendered *= 1000 # convert meter to mm
        render.scene.clear_geometry()

        # rendering object masks #
        obj_masks = {}
        for source_obj in objects:
            # add geometry and set color (target object as white / others as black)
            for target_obj in objects:
                target_obj = copy.deepcopy(target_obj)
                color = [1,1,1] if source_obj.obj_name == target_obj.obj_name else [0,0,0]
                target_obj.obj_mesh.paint_uniform_color(color)
                render.scene.add_geometry("mask_{}_to_{}".format(
                                                source_obj.obj_name, target_obj.obj_name), 
                                        target_obj.obj_mesh, obj_mtl,                              
                                        add_downsampled_copy_for_fast_rendering=True)
            # render mask as RGB
            mask_obj = render.render_to_image()
            mask_obj = np.array(mask_obj)
            obj_masks[source_obj.obj_name] = mask_obj.copy()
            render.scene.clear_geometry()


        depth_captured = cv2.imread(self.depth_path, -1)
        depth_captured = cv2.resize(depth_captured, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_captured = np.float32(depth_captured) * self.scene_camera_info[str(self.image_num_lists[self.current_image_idx])]["depth_scale"] 
        valid_depth_mask = np.array(depth_captured > 200, dtype=bool)

        rgb_img = cv2.imread(self.rgb_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        diff_vis = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        ########################################
        # calculate depth difference with mask #
        # depth_diff = depth_cap - depth_ren   #
        ########################################
        texts = []
        is_oks = []
        # self.H, self.W, _ = diff_vis.shape
        self.icx, self.icy = self.W / 2, self.H / 2
        self.scale_factor = 1
        self.depth_diff_means = {}
        amodal_masks = []
        bboxes = []
        cmap = matplotlib.cm.get_cmap('hsv')
        for i, (obj_name, obj_mask) in enumerate(obj_masks.items()):
            obj_mask = np.where(obj_mask > 125, 255, 0).astype(np.uint8)[..., 0]
            valid_mask = obj_mask * copy.deepcopy(valid_depth_mask)
            valid_mask = np.array(valid_mask > 0, dtype=bool).astype(np.uint8)

            depth_captured_obj = depth_captured.copy()
            depth_rendered_obj = depth_rendered.copy()

            depth_diff = depth_captured_obj - depth_rendered_obj
            inlier_mask = np.abs(np.copy(depth_diff)) < 100
            depth_diff = depth_diff * valid_mask
            depth_diff_abs = np.abs(np.copy(depth_diff))
            
            if np.sum(inlier_mask) == 0:
                depth_diff = np.ones_like(depth_diff) * 1000
                depth_diff_abs = np.ones_like(depth_diff_abs) * 1000

            delta_1 = 5 # mm, threshold for high-quality annotation 
            delta_2 = 15 ## mm, threshold for ok-ish annotation
            below_delta_1 = valid_mask * (depth_diff_abs < delta_1)
            below_delta_2 = valid_mask * (depth_diff_abs < delta_2) * (depth_diff_abs > delta_1)
            above_delta = valid_mask * (depth_diff_abs > delta_2)
            below_delta_1_vis = (below_delta_1).astype(np.uint8)
            below_delta_2_vis = (below_delta_2).astype(np.uint8)
            above_delta_vis = (above_delta).astype(np.uint8)
            depth_diff_mean = np.sum(depth_diff[valid_mask > 0]) / np.sum(valid_mask)
            depth_diff_vis = np.stack([below_delta_2_vis, below_delta_1_vis, above_delta_vis], axis=-1) * 255

            try:
                diff_vis[valid_mask>0] = depth_diff_vis[valid_mask>0]
            except Exception as e:
                print(e)
                self._on_error("Object {} is out of camera view or too far from point cloud.".format(obj_name))
                continue
            text = "{}_{}".format(int(obj_name.split("_")[1]), int(obj_name.split("_")[2]))
            texts.append(text)
            ys, xs = valid_mask.nonzero()
            bbox = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

            self.depth_diff_means[obj_name] = abs(depth_diff_mean)
            ok_delta = self.ok_delta
            ok_delta *= camera_idx_to_thresh_factor[self.current_image_idx % 4]
            is_ok = abs(depth_diff_mean) < ok_delta
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv2.rectangle(diff_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

            is_oks.append(is_ok)
            amodal_masks.append(obj_mask)
            bboxes.append(bbox)
        # Pre-compute the colors and the text overlay positions once, outside the loop
        colors = [np.array(cmap(i / len(amodal_masks))[:3]) * 255 for i in range(len(amodal_masks))]
        text_positions = [(bbox[0], bbox[1]) for bbox in bboxes]

        # Create a copy of the original image to apply masks
        mask_img = np.zeros_like(rgb_img)
        # draw amodal masks
        for amodal_mask, text_position, color in zip(amodal_masks, text_positions, colors):
            mask_img[amodal_mask > 0] = color
            cv2.putText(mask_img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(mask_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

        mask_img = cv2.addWeighted(rgb_img, 0.5, mask_img, 1.0, 0)
        self.mask_img = mask_img
        diff_img = cv2.addWeighted(rgb_img, 0.5, diff_vis, 0.8, 0)
        self.diff_img = diff_img
        self._update_vis_img(rgb_img, diff_img, mask_img)

    def _on_error(self, err_msg):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(err_msg))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_show_coord_frame(self, show):
        self.settings.show_coord_frame = show
        self._apply_settings()

    def _on_show_mesh_names(self, show):
        self.settings.show_mesh_names = show
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self._log.text = "\t Highlighting the object to annotate."
            self.window.set_needs_layout()
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
            self.settings.annotation_active_obj_material.base_color = [0.3, 0.9, 0.3, 1.0]
        elif not light:
            self._log.text = "\t Not highlighting the object to annotate."
            self.window.set_needs_layout()
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]
            self.settings.annotation_active_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        if self._annotation_scene is None: 
            self._on_error("Select the annotation object file. (error at _on_highlight_obj)")
            return
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(mesh.obj_name, self.settings.annotation_obj_material)
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.modify_geometry_material(active_obj.obj_name, self.settings.annotation_active_obj_material)


    def _on_transparency(self, transparency): 
        
        self._log.text = "\t Adjusting transparency."
        self.window.set_needs_layout()
        self.settings.transparency = transparency
        if self._annotation_scene is None:
            return
        self.settings.annotation_obj_material.base_color = [0.9, 0.3 + 0.6*transparency, 0.3 + 0.6*transparency, 1]
        self.settings.annotation_active_obj_material.base_color = [0.3 + 0.6*transparency, 0.9, 0.3 + 0.6*transparency, 1]

        objects = self._annotation_scene.get_objects()
        for obj in objects:
            self._scene.scene.remove_geometry(obj.obj_name)
            self._scene.scene.add_geometry(obj.obj_name, obj.obj_geometry,
                                            self.settings.annotation_obj_material,
                                            add_downsampled_copy_for_fast_rendering=True)

        active_obj = objects[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)                       
        self._apply_settings()


    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_responsiveness(self, responsiveness):
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")
        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("6D Object Pose Annotator by Seunghyeok Back (GIST, KIMM)\nshback@kimm.re.kr"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _obj_instance_count(self, mesh_to_add, meshes):
        types = [i[:-2] for i in meshes]  # remove last 3 character as they present instance number (OBJ_INSTANCE)
        equal_values = [i for i in range(len(types)) if types[i] == mesh_to_add]
        count = 1
        if len(equal_values):
            indices = np.array(meshes)
            indices = indices[equal_values]
            indices = [int(x[-1]) for x in indices]
            count = max(indices) + 1
            # TODO change to fill the numbers missing in sequence
        return count 

    def _add_mesh(self):
        if self._annotation_scene is None: 
            self._on_error("Select the file to annotate. (error at _add_mesh)") 
            return

        mesh_name_to_add = f'obj_{self._meshes_available.int_value:06}'
        if not os.path.exists(os.path.join(self.scenes.objects_path, mesh_name_to_add + '.ply')):
            self._on_error("The object does not exist. (error at _add_mesh)")
            return

        self._log.text = "\t Adding object to annotate."
        self.window.set_needs_layout()
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]
        object_geometry = o3d.io.read_point_cloud(
            os.path.join(self.scenes.objects_path, mesh_name_to_add + '.ply'))
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000)  # convert mm to meter
        object_mesh = o3d.io.read_triangle_mesh(
                os.path.join(self.scenes.mesh_path, mesh_name_to_add + '.obj'))
        object_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(object_mesh.vertices) / 1000)  # convert mm to meter
        init_trans = np.identity(4)
        center = self._annotation_scene.annotation_scene.get_center()
        center[2] -= 0.2
        init_trans[0:3, 3] = center
        # object_geometry.transform(init_trans)
        new_mesh_instance = self._obj_instance_count(mesh_name_to_add, meshes)
        new_mesh_name = mesh_name_to_add + '_' + str(new_mesh_instance)
        self._scene.scene.add_geometry(new_mesh_name, object_geometry, self.settings.annotation_obj_material,
                                       add_downsampled_copy_for_fast_rendering=True)
        self._annotation_scene.add_obj(object_geometry, object_mesh, new_mesh_name, new_mesh_instance, transform=init_trans)
        if self.settings.show_mesh_names:
            self.mesh_names.append(self._scene.add_3d_label(center, f"{new_mesh_name}"))

        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = len(meshes) - 1
        self._annotation_changed = True

    def _remove_mesh(self):
        if self._annotation_scene is None: 
            self._on_error("Select the file to annotate. (error at _remove_mesh)")
            return
        if not self._annotation_scene.get_objects():
            self._on_error("There is no object to remove. (error at _remove_mesh)")
            return
        self._log.text = "\t Removing object to annotate."
        self.window.set_needs_layout()
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(self._meshes_used.selected_index)  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()
        self._annotation_changed = True

    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                      cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def _update_vis_img(self, rgb_img, diff_img, mask_img):
        
        width = 512
        ratio = width / self.W
        rgb_img = cv2.resize(rgb_img, (width, int(self.H*ratio)))
        diff_img = cv2.resize(diff_img, (width, int(self.H*ratio)))
        mask_img = cv2.resize(mask_img, (width, int(self.H*ratio)))
        
        vis_img = np.vstack([rgb_img, diff_img, mask_img])
        _vis_img = o3d.geometry.Image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        self._vis_img_proxy.set_widget(gui.ImageWidget(_vis_img))
        self._log.text = "\t Updated visualization images."
        self.window.set_needs_layout()   


    def scene_load(self, scenes_path, scene_num, image_num):

        self._annotation_changed = False
        self._scene.scene.clear_geometry()
        geometry = None

        scene_path = os.path.join(scenes_path, f'{scene_num:06}')
        camera_params_path = os.path.join(scene_path, 'scene_camera.json'.format(self.current_scene_idx)) 
        with open(camera_params_path) as f:
            self.scene_camera_info = json.load(f)
            cam_K = self.scene_camera_info[str(image_num)]['cam_K']
            self.cam_K = np.array(cam_K).reshape((3, 3))
            depth_scale = self.scene_camera_info[str(image_num)]['depth_scale']
        if image_num < 0:
            self.rgb_path = os.path.join(scene_path, 'rgb', f'{image_num:07}.png')
            self.depth_path = os.path.join(scene_path, 'depth', f'{image_num:07}.png')
        else:
            self.rgb_path = os.path.join(scene_path, 'rgb', f'{image_num:06}.png')
            self.depth_path = os.path.join(scene_path, 'depth', f'{image_num:06}.png')
        if not os.path.exists(self.rgb_path):
            self.rgb_path = os.path.join(scene_path, 'rgb', f'{image_num:06}.jpg')

        self.rgb_img = cv2.imread(self.rgb_path)
        depth_img = cv2.imread(self.depth_path, -1)
        depth_img = np.float32(depth_img) / 1000 * depth_scale
        self.H, self.W, _ = self.rgb_img.shape
        self.H, self.W = self.H // 4, self.W // 4
        rgb_img = self.rgb_img.copy()
        diff_img = np.zeros_like(rgb_img)
        mask_img = np.zeros_like(rgb_img)
        self._update_vis_img(rgb_img, diff_img, mask_img)

        geometry = self._make_point_cloud(self.rgb_img, depth_img, self.cam_K)

        if geometry is not None:
            print("[Info] Successfully read scene ", scene_num)
            if not geometry.has_normals():
                geometry.estimate_normals()
            geometry.normalize_normals()
        else:
            print("[WARNING] Failed to read points")
        self._scene.scene.add_geometry("annotation_scene", geometry, self.settings.scene_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        self.bounds = geometry.get_axis_aligned_bounding_box()
        self._on_initial_viewpoint()

        self._annotation_scene = AnnotationScene(geometry, scene_num, image_num)
        self._meshes_used.set_items([])  # clear list from last loaded scene

        # load values if an annotation already exists
        scene_gt_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", 'scene_gt.json')
        if os.path.exists(scene_gt_path):
            with open(scene_gt_path) as scene_gt_file:
                try:
                    data = json.load(scene_gt_file)
                except json.decoder.JSONDecodeError:
                    self._on_error("Failed to load annotation file. (error at scene_load)")
                    return
                if str(image_num) in data.keys():
                    scene_data = data[str(image_num)]
                    active_meshes = list()
                    sorted_scene_data = sorted(scene_data, key=lambda d: int(d['obj_id']))
                    for i, obj in enumerate(sorted_scene_data):
                        # add object to annotation_scene object
                        obj_geometry = o3d.io.read_point_cloud(
                            os.path.join(self.scenes.objects_path, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
                        obj_geometry.points = o3d.utility.Vector3dVector(
                            np.array(obj_geometry.points) / 1000)  # convert mm to meter
                        obj_mesh = o3d.io.read_triangle_mesh(
                            os.path.join(self.scenes.mesh_path, 'obj_' + f"{int(obj['obj_id']):06}" + '.obj'))
                        obj_mesh.vertices = o3d.utility.Vector3dVector(
                            np.array(obj_mesh.vertices) / 1000)  # convert mm to meter
                        
                        model_name = 'obj_' + f'{ + obj["obj_id"]:06}'
                        if "inst_id" in obj.keys():
                            obj_instance = int(obj["inst_id"])
                        else:
                            obj_instance = self._obj_instance_count(model_name, active_meshes)
                        obj_name = model_name + '_' + str(obj_instance)
                        translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
                        orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
                        transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
                        transform_cam_to_obj = np.concatenate(
                            (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform

                        self._annotation_scene.add_obj(obj_geometry, obj_mesh, obj_name, obj_instance, transform_cam_to_obj)
                        # obj_geometry.transform(transform_cam_to_obj)
                        self._scene.scene.add_geometry(obj_name, obj_geometry, self.settings.annotation_obj_material,
                                                        add_downsampled_copy_for_fast_rendering=True)
                        active_meshes.append(obj_name)
                    self._meshes_used.set_items(active_meshes)


        self._update_scene_numbers()
        self._validate_anno()

        self.update_scene_obj_info_table()

        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        current_scene_num = self.scene_num_lists[self.current_scene_idx]

    def update_obj_list(self):
        model_names = self.load_model_names()
        max_obj_id = max([int(x.split('_')[-1]) for x in model_names])
        self._meshes_available.set_limits(1, max_obj_id)

    def load_model_names(self):
        self.obj_ids = sorted([int(os.path.basename(x)[5:-4]) for x in glob.glob(self.scenes.objects_path + self.spl + '*.ply')])
        model_names = ['obj_' + f'{ + obj_id:06}' for obj_id in self.obj_ids]
        return model_names

    def _on_initial_viewpoint(self):
        if self.bounds is None:
            self._on_error("No point cloud loaded. (error at _on_initial_viewpoint)")
            return
        self._log.text = "\t Moving to initial viewpoint."
        self.window.set_needs_layout()

        intrinsic = np.array(self.cam_K).reshape((3, 3))
        extrinsic = np.eye(4)
        self._scene.setup_camera(intrinsic, extrinsic, int(self.W*4), int(self.H*4), self.bounds)
        center = [0, 0, 1]  # look_at target
        eye = [0, 0, -0.5]  # camera position
        up = [0, -1, 0]  # camera orientation
        self._scene.look_at(center, eye, up)

        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _check_changes(self):
        if self._annotation_changed:
            self._on_error("Annotation has been changed. Save the annotation first. (error at _check_changes)")
            self._annotation_changed = False
            return True
        else:
            return False

    def _on_next_scene(self):
        if self._check_changes():
            return
        if self.current_scene_idx is None:
            self._on_error("Select the annotation object file. (error at _on_next_scene)")
            return
        if self.current_scene_idx >= len(self.scene_num_lists) - 1:
            self._on_error("Next scene does not exist.")
            return
        self._log.text = "\t Moving to the next scene."
        self.window.set_needs_layout()
        self.current_scene_idx += 1
        self.scene_load(self.scenes.scenes_path, self.scene_num_lists[self.current_scene_idx], -4)  # open next scene on the first image

    def _on_previous_scene(self):
        if self._check_changes():
            return
        if self.current_scene_idx is None:
            self._on_error("Select the annotation object file. (error at _on_previous_scene)")
            return
        if self.current_scene_idx <= 0:
            self._on_error("Previous scene does not exist.")
            return
        self.current_scene_idx -= 1
        self._log.text = "\t Moving to the previous scene."
        self.window.set_needs_layout()
        self.scene_load(self.scenes.scenes_path, self.scene_num_lists[self.current_scene_idx], -4)  # open next scene on the first image

    def _on_change_image(self):
        if self._check_changes():
            return
        try:
            if self.image_number_edit.int_value not in self.image_num_lists:
                self._on_error("The image number does not exist. (error at _on_change_image)")
                return
        except AttributeError:
            self._on_error("Select the annotation object file. (error at _on_change_image)")
            return
        self._log.text = "\t Changing the image."
        self.window.set_needs_layout()
        self.current_image_idx = self.image_num_lists.index(self.image_number_edit.int_value)
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num, self.image_num_lists[self.current_image_idx])
        self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) # 25% complete
        self._progress_str.text = "Progress: {:.1f}% [{}/{}]".format(
            100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
            self.current_image_idx + 1, len(self.image_num_lists))

    def _on_next_image(self):
        if self._check_changes():
            return
        if self.current_image_idx is None:
            self._on_error("Select the annotation object file. (error at _on_next_image)")
            return
        if self.current_image_idx  >= len(self.image_num_lists) - 1:
            self._on_error("Next point cloud does not exist. (error at _on_next_image)")
            return
        self._log.text = "\t Moving to the next point cloud."
        self.window.set_needs_layout()
        self.current_image_idx += 1
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num, self.image_num_lists[self.current_image_idx])
        self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) # 25% complete
        self._progress_str.text = "Progress: {:.1f}% [{}/{}]".format(
            100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
            self.current_image_idx + 1, len(self.image_num_lists))

    def _on_previous_image(self):
        if self._check_changes():
            return
        if self.current_image_idx is None:
            self._on_error("Select the annotation object file. (error at _on_previous_image)")
            return
        if self.current_image_idx < -4:
            self._on_error("Previous point cloud does not exist. (error at _on_previous_image)")
            return
        self._log.text = "\t Moving to the previous point cloud."
        self.window.set_needs_layout()
        self.current_image_idx -= 1
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num, self.image_num_lists[self.current_image_idx])
        self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) 
        self._progress_str.text = "Progress: {:.1f}% [{}/{}]".format(
            100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
            self.current_image_idx + 1, len(self.image_num_lists))


def main():


    gui.Application.instance.initialize()
    w = AppWindow(1920, 1080)
    gui.Application.instance.run()


if __name__ == "__main__":

    main()
