import argparse

from manager import TeleopManager

# create a tv.step() thread and request image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument(
        "--task_name", type=str, default="default_task", help="Name of the task"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--robot", default="g1", help="Use g1 controllers")
    parser.add_argument(
        "--hand-type",
        default="dex3",
        choices=["dex3", "dex1"],
        help="Select hand backend for G1.",
    )
    parser.add_argument(
        "--dex1-control-mode",
        default="gesture_open_close",
        choices=["gesture_open_close"],
        help="Dex1 control mode.",
    )
    parser.add_argument("--dex1-open-q", type=float, default=0.0)
    parser.add_argument("--dex1-close-q", type=float, default=5.5)
    parser.add_argument("--dex1-fist-threshold", type=float, default=0.85)
    parser.add_argument("--dex1-open-threshold", type=float, default=0.55)
    parser.add_argument(
        "--dex1-fist-polarity",
        default="high_is_fist",
        choices=["high_is_fist", "low_is_fist"],
        help="How to interpret retarget qpos scalar as fist/open.",
    )
    parser.add_argument(
        "--dex1-debug",
        action="store_true",
        help="Enable throttled Dex1 fist/open telemetry logs.",
    )
    args = parser.parse_args()

    manager = TeleopManager(
        task_name=args.task_name,
        robot=args.robot,
        debug=args.debug,
        hand_type=args.hand_type,
        dex1_control_mode=args.dex1_control_mode,
        dex1_open_q=args.dex1_open_q,
        dex1_close_q=args.dex1_close_q,
        dex1_fist_threshold=args.dex1_fist_threshold,
        dex1_open_threshold=args.dex1_open_threshold,
        dex1_fist_polarity=args.dex1_fist_polarity,
        dex1_debug=args.dex1_debug,
    )
    manager.start_processes()
    # TODO: run in two separate terminals for debuggnig
    manager.run_command_loop()
