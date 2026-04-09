import threading
import time
import tkinter as tk
from tkinter import ttk

import crocoddyl
import numpy as np
import pinocchio
from param_parsers import ParamParser
from visualizer import add_sphere_to_viewer


class GUI:
    def __init__(
        self,
        rmodel: pinocchio.Model,
        ref_frame_id: int,
        cmodel: pinocchio.GeometryModel,
        moving_geom: int,
        frame_placement_residual: crocoddyl.ResidualModelFramePlacement,
    ):
        self.rmodel = rmodel
        self.ref_frame_id = ref_frame_id
        self.cmodel = cmodel
        self.moving_geom = moving_geom
        self.frame_placement_residual = frame_placement_residual
        self.stop_requested = False

        self.mutex = threading.Lock()

    def update_geom(self):
        with self.mutex:
            self.cmodel.geometryObjects[
                self.moving_geom
            ].placement.translation = np.array(
                [
                    float(self.entry_geom_x.get()),
                    float(self.entry_geom_y.get()),
                    float(self.entry_geom_z.get()),
                ]
            )
            self.cmodel.geometryObjects[
                self.moving_geom
            ].placement.rotation = pinocchio.rpy.rpyToMatrix(
                float(self.entry_geom_roll.get()),
                float(self.entry_geom_pitch.get()),
                float(self.entry_geom_yaw.get()),
            )

    def update_reference(self):
        ref = pinocchio.SE3(
            pinocchio.rpy.rpyToMatrix(
                float(self.entry_ref_roll.get()),
                float(self.entry_ref_pitch.get()),
                float(self.entry_ref_yaw.get()),
            ),
            np.array(
                [
                    float(self.entry_ref_x.get()),
                    float(self.entry_ref_y.get()),
                    float(self.entry_ref_z.get()),
                ]
            ),
        )
        with self.mutex:
            self.frame_placement_residual.reference = ref
            self.rmodel.frames[self.ref_frame_id].placement = ref

            print(self.frame_placement_residual)

    def run(self):
        root = tk.Tk()
        root.title("3D Transformations")

        frame_geom = ttk.LabelFrame(root, text="Obstacle Transformation")
        frame_geom.grid(row=0, column=0, padx=10, pady=10)

        T = self.cmodel.geometryObjects[self.moving_geom].placement.translation
        ttk.Label(frame_geom, text="X:").grid(row=0, column=0)
        self.entry_geom_x = ttk.Entry(frame_geom)
        self.entry_geom_x.insert(0, str(T[0]))
        self.entry_geom_x.grid(row=0, column=1)

        ttk.Label(frame_geom, text="Y:").grid(row=1, column=0)
        self.entry_geom_y = ttk.Entry(frame_geom)
        self.entry_geom_y.insert(0, str(T[1]))
        self.entry_geom_y.grid(row=1, column=1)

        ttk.Label(frame_geom, text="Z:").grid(row=2, column=0)
        self.entry_geom_z = ttk.Entry(frame_geom)
        self.entry_geom_z.insert(0, str(T[2]))
        self.entry_geom_z.grid(row=2, column=1)

        R = self.cmodel.geometryObjects[self.moving_geom].placement.rotation
        rpy = pinocchio.rpy.matrixToRpy(R)
        ttk.Label(frame_geom, text="Roll:").grid(row=3, column=0)
        self.entry_geom_roll = ttk.Entry(frame_geom)
        self.entry_geom_roll.insert(0, str(rpy[0]))
        self.entry_geom_roll.grid(row=3, column=1)

        ttk.Label(frame_geom, text="Pitch:").grid(row=4, column=0)
        self.entry_geom_pitch = ttk.Entry(frame_geom)
        self.entry_geom_pitch.insert(0, str(rpy[1]))
        self.entry_geom_pitch.grid(row=4, column=1)

        ttk.Label(frame_geom, text="Yaw:").grid(row=5, column=0)
        self.entry_geom_yaw = ttk.Entry(frame_geom)
        self.entry_geom_yaw.insert(0, str(rpy[2]))
        self.entry_geom_yaw.grid(row=5, column=1)

        ttk.Button(frame_geom, text="Update Obstacle", command=self.update_geom).grid(
            row=6, column=1, padx=10, pady=10
        )

        frame_ref = ttk.LabelFrame(root, text="Reference Transformation")
        frame_ref.grid(row=1, column=0, padx=10, pady=10)

        T = self.frame_placement_residual.reference.translation
        ttk.Label(frame_ref, text="X:").grid(row=0, column=0)
        self.entry_ref_x = ttk.Entry(frame_ref)
        self.entry_ref_x.insert(0, str(T[0]))
        self.entry_ref_x.grid(row=0, column=1)

        ttk.Label(frame_ref, text="Y:").grid(row=1, column=0)
        self.entry_ref_y = ttk.Entry(frame_ref)
        self.entry_ref_y.insert(0, str(T[1]))
        self.entry_ref_y.grid(row=1, column=1)

        ttk.Label(frame_ref, text="Z:").grid(row=2, column=0)
        self.entry_ref_z = ttk.Entry(frame_ref)
        self.entry_ref_z.insert(0, str(T[2]))
        self.entry_ref_z.grid(row=2, column=1)

        R = self.frame_placement_residual.reference.rotation
        rpy = pinocchio.rpy.matrixToRpy(R)
        ttk.Label(frame_ref, text="Roll:").grid(row=3, column=0)
        self.entry_ref_roll = ttk.Entry(frame_ref)
        self.entry_ref_roll.insert(0, str(rpy[0]))
        self.entry_ref_roll.grid(row=3, column=1)

        ttk.Label(frame_ref, text="Pitch:").grid(row=4, column=0)
        self.entry_ref_pitch = ttk.Entry(frame_ref)
        self.entry_ref_pitch.insert(0, str(rpy[1]))
        self.entry_ref_pitch.grid(row=4, column=1)

        ttk.Label(frame_ref, text="Yaw:").grid(row=5, column=0)
        self.entry_ref_yaw = ttk.Entry(frame_ref)
        self.entry_ref_yaw.insert(0, str(rpy[2]))
        self.entry_ref_yaw.grid(row=5, column=1)

        ttk.Button(
            frame_ref, text="Update Reference", command=self.update_reference
        ).grid(row=6, column=1, padx=10, pady=10)

        ttk.Button(root, text="Stop", command=self.stop_request).grid(
            row=2, column=0, padx=10, pady=10
        )
        root.wm_attributes("-topmost", True)

        self.root = root
        root.mainloop()

    def start(self):
        # Run the GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.run)
        self.gui_thread.daemon = True
        self.gui_thread.start()

    def stop_request(self):
        self.stop_requested = True

    def stop(self):
        self.root.quit()
        self.gui_thread.join()

    def is_running(self):
        return self.gui_thread.is_alive()


def simulation_loop(
    ocp: crocoddyl.SolverAbstract,
    rmodel: pinocchio.Model,
    ref_frame_id: int,
    cmodel: pinocchio.GeometryModel,
    moving_geom: int,
    frame_placement_residual: crocoddyl.ResidualModelFramePlacement,
    pp: ParamParser,
    vis: pinocchio.visualize.BaseVisualizer,
):
    rdata = pinocchio.Data(rmodel)
    vis_id = frame_placement_residual.id  # rmodel.getFrameId("panda2_rightfinger")

    gui = GUI(rmodel, ref_frame_id, cmodel, moving_geom, frame_placement_residual)
    gui.start()

    try:
        dt = pp.get_dt()

        xs = [pp.get_X0()] * (pp.get_T() + 1)
        us = ocp.problem.quasiStatic(xs[:-1])

        ok = ocp.solve(xs, us, 1000)
        print(ok, ocp.cost)
        assert ok
        xs = ocp.xs.copy()
        us = ocp.us.copy()

        i = 0
        input()
        while not gui.stop_requested:
            i = i + 1
            t0 = time.time()
            with gui.mutex:
                ok = ocp.solve(xs, us, 100)
            t1 = time.time()
            print(i, t1 - t0)
            xs = ocp.xs.copy()
            us = ocp.us.copy()
            q = xs[1][:7]
            if not ok:
                print("Failed to solve the problem")
            # print(
            # "u: ",
            # np.array2string(
            # np.array(us), precision=2, separator=", ", suppress_small=True
            # ),
            # )
            # print(
            # "q: ",
            # np.array2string(
            # np.array(xs)[:, :7],
            # precision=2,
            # separator=", ",
            # suppress_small=True,
            # ),
            # )
            # print(
            # "q1: ",
            # np.array2string(
            # xs[1][:7], precision=2, separator=", ", suppress_small=True
            # ),
            # )
            # print(
            # "diff: ",
            # np.array2string(
            # xs[1][:7] - xs[0][:7],
            # precision=2,
            # separator=", ",
            # suppress_small=True,
            # ),
            # )
            # for q0, q1 in zip(xs[:-1], xs[1:]):
            # print(
            # "diff: ",
            # np.array2string(
            # q1[:7] - q0[:7],
            # precision=2,
            # separator=", ",
            # suppress_small=True,
            # ),
            # )

            # Shift the trajectory
            xs[:-1] = xs[1:]
            us[:-1] = us[1:]
            xs[-1] = xs[-2]
            us[-1] = np.zeros_like(us[-1])
            ocp.problem.x0 = xs[0]

            # Update the visualizer
            sleep_time = dt - (time.time() - t0)
            if sleep_time < 0:
                print("Warning: the loop is running slower than real time")
            time.sleep(max(0, sleep_time))
            vis.display(q)

            for k, x in enumerate(ocp.xs):
                qq = np.array(x[:7].tolist())
                pinocchio.forwardKinematics(rmodel, rdata, qq)
                pinocchio.updateFramePlacement(rmodel, rdata, vis_id)
                add_sphere_to_viewer(
                    vis,
                    f"colmpc{k}",
                    2e-2,
                    rdata.oMf[vis_id].translation,
                    color=100000,
                )
    finally:
        gui.stop()
