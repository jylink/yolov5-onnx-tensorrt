import tensorrt as trt
import sys
import argparse

"""
takes in onnx model
converts to tensorrt
"""

if __name__ == '__main__':

    desc = 'compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', help='onnx file')
    parser.add_argument('--fp', type=int, default=16, help='floating point precision. 16 or 32')
    parser.add_argument('--out', type=str, default='', help='name of trt output file')
    opt = parser.parse_args()
    
    batch_size = 1
    model = opt.model
    fp = opt.fp
    output = opt.out if opt.out else opt.model.replace('.onnx', '.trt')
    assert fp in (16, 32)
    
    logger = trt.Logger(trt.Logger.WARNING)
    print('trt version', trt.__version__)
    assert trt.__version__[0] >= '7'
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        if fp == 16:
            builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print('ERROR', parser.get_error(error))
        
        # if your onnx has a dynamic input...
        # network.get_input(0).shape = [1, 3, 352, 608]
        
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
        print('Done')