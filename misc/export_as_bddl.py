
def _normalize_object_name(name):
    """Convert object name to BDDL-compatible format."""
    return name.replace(' ', '_')

def _create_bddl_regions():
    """Generate the regions section of the BDDL file."""
    return """(:regions
        (bin_region
            (:target floor)
            (:ranges (
                (-0.01 0.25 0.01 0.27)
                )
            )
        )
        (target_object_region
            (:target floor)
            (:ranges (
                (-0.145 -0.265 -0.095 -0.215)
                )
            )
        )
    )"""

def _create_bddl_objects(object_names):
    """Generate the objects section of the BDDL file."""
    object_entries = []
    for name in object_names:
        normalized_name = _normalize_object_name(name)
        prefixed_name = f"objaverse_{normalized_name}"
        object_entries.append(f"    {prefixed_name}_1 - {prefixed_name}")
    
    return "(:objects\n" + "\n".join(object_entries) + "\n    )"

def _create_bddl_init_state(object_names):
    """Generate the init state section of the BDDL file."""
    init_entries = []
    for name in object_names:
        normalized_name = _normalize_object_name(name)
        prefixed_name = f"objaverse_{normalized_name}"
        init_entries.append(f"    (On {prefixed_name}_1 floor_target_object_region)")
    
    return "(:init\n" + "\n".join(init_entries) + "\n    )"

def export_as_bddl_file(object_names, target_object_name, bddl_file_name):
    """Generate and save a BDDL file for the given objects and target."""
    # Prepare target object information
    normalized_target = _normalize_object_name(target_object_name)
    target_instance = f"objaverse_{normalized_target}_1"
    
    # Generate BDDL sections
    regions_section = _create_bddl_regions()
    objects_section = _create_bddl_objects(object_names)
    init_state_section = _create_bddl_init_state(object_names)
    
    # Construct the complete BDDL content
    bddl_content = f"""(define (problem LIBERO_Floor_Manipulation)
    (:domain robosuite)
    (:language pick up {target_object_name})
    {regions_section}

    (:fixtures
        floor - floor
    )

    {objects_section}

    (:obj_of_interest
        {target_instance}
    )

    {init_state_section}

    (:goal
        (And (Grasped {target_instance}))
    )
)
"""
    
    # Write to file
    with open(bddl_file_name, 'w') as f:
        f.write(bddl_content)
    
    return bddl_content