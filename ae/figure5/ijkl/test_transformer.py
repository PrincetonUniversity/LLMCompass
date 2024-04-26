from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import system_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="use roofline")
    args = parser.parse_args()

    bs = 8
    s = 2048
    if args.init:
        print("Initial computation")
        if args.simgpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16"]
            # from design_space_exploration.dse import read_architecture_template, template_to_system
            # arch_specs = read_architecture_template("configs/template.json")
            # A100_system = template_to_system(arch_specs)
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = "transformer_A100_roofline.csv"
            else:
                model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
                file_name = "transformer_A100_sim.csv"
        if args.simtpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformer_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformer_TPUv3_sim.csv"
        if args.gpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            model.run_on_gpu()
    else:
        print("Auto-regression")
        output_token_length = 1024
        if args.simgpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = "transformerAR_A100_roofline.csv"
            else:
                model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
                file_name = "transformerAR_A100_sim.csv"
        if args.simtpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformerAR_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformerAR_TPUv3_sim.csv"
        if args.gpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            model.run_on_gpu()
    with open(f"ae/figure5/ijkl/{file_name}", "w") as f:
        if args.roofline:
            f.write(model.roofline_log)
        else:
            f.write(model.simluate_log)
