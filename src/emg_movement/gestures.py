# NinaPro gesture names (1-indexed label -> name; index 0 = label 1, etc.)

ALL_GESTURES = [
    "index_flexion",
    "index_extension",
    "middle_flexion",
    "middle_extension",
    "ring_flexion",
    "ring_extension",
    "little_flexion",
    "little_extension",
    "thumb_adduction",
    "thumb_abduction",
    "thumb_flexion",
    "thumb_extension",
    "thumb_up",
    "index_middle_extension_others_flexed",
    "ring_little_flexion_others_extended",
    "thumb_opposition",
    "finger_abduction",
    "fist",
    "pointing_index",
    "finger_adduction",
    "wrist_supination_middle_axis",
    "wrist_pronation_middle_axis",
    "wrist_supination_little_axis",
    "wrist_pronation_little_axis",
    "wrist_flexion",
    "wrist_extension",
    "wrist_radial_deviation",
    "wrist_ulnar_deviation",
    "wrist_extension_closed_hand",
    "large_diameter_grasp",
    "small_diameter_grasp",
    "fixed_hook_grasp",
    "index_extension_grasp",
    "medium_wrap",
    "ring_grasp",
    "prismatic_four_finger_grasp",
    "stick_grasp",
    "writing_tripod_grasp",
    "power_sphere_grasp",
    "three_finger_sphere_grasp",
    "precision_sphere_grasp",
    "tripod_grasp",
    "prismatic_pinch_grasp",
    "tip_pinch_grasp",
    "quadrupod_grasp",
    "lateral_grasp",
    "parallel_extension_grasp",
    "extension_type_grasp",
    "power_disk_grasp",
    "open_bottle_tripod_grasp",
    "turn_screw_grasp",
    "cut_something_index_grasp",
]


def label_to_gesture_name(label: int) -> str:
    """Map 1-indexed NinaPro label to gesture name. Label 0 = 'rest'."""
    if label == 0:
        return "rest"
    if 1 <= label <= len(ALL_GESTURES):
        return ALL_GESTURES[label - 1]
    return f"label_{label}"
