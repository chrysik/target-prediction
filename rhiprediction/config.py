"""Config file."""
raw_data_folder = "raw_data"

df_raw_types = {"datetime_cols":
                ["when", "expected_start", "start_process",
                 "start_subprocess1", "start_critical_subprocess1",
                 "predicted_process_end", "process_end",
                 "subprocess1_end", "reported_on_tower", "opened"],
                "int_cols":
                    ["unnamed_0", "groups", "tracking", "unnamed_7",
                     "human_measure", "expected_factor_x"],
                "float_cols":
                    ["crystal_weight", "previous_factor_x", "first_factor_x",
                     "expected_final_factor_x", "final_factor_x",
                     "previous_adamantium", "unnamed_17",
                     "chemical_x", "raw_kryptonite", "argon", "pure_seastone",
                     "etherium_before_start"],
                "categ_cols":
                    ["place", "super_hero_group", "crystal_type",
                     "crystal_supergroup", "cycle", "human_behavior_report",
                     "tracking_times"]
                }

df_target_types = None
