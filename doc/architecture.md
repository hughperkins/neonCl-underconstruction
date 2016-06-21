# architecture

## API

winogradcl.api:
  - shuffle
  - fprop
  - bprop_gradI
  - bprop_gradW

## NeonCl, Current

winogradcl.layers.layer:
    Convolution(Layer) class:
       - contains buffers
       - calcs buffer sizes
       configure(in_obj):
          self.nglayer = nervanagpu.conv_layer()
       fprop(inputs):
          nervanagpu.fprop_conv(self.nglayer, inputs, self.W, self.outputs)
       - calls into:
          - nervanagpu.fprop_conv
          - nervanagpu.bprop_conv
          - nervanagpu.update_conv

winogradcl.backends.nervanagpu:
    NervanaGpu class:
      __init__:
       - sets up cl queue and cl context
       - owns and manages scratch buffer
       - calls 
      conv_layer(*dimensions):
         return winogradcl.backends.layer_gpu.ConvLayer(*dimensions)
      fprop_conv(layer, I, F, O):
         layer.fprop_kernels_bind_params(I,F,O)
         layer.fprop_kernels.execute()

winogradcl.backends.layer_gpu.py:
   Layer(object)
   ConvLayer(Layer)
      __init__(*dimensions):
         self.fprop_kernels = winogradcl.backends.convolution.FpropCuda(N, C, K, D, H, W, T R, S M, P, Q,
                       pad_d, pad_h, pad_w, str_d, str_h, str_w)
         self.bprop_kernels = winogradcl.backends.convolution.BpropCuda(N, C, K, D, H, W, T, R, S, M, P, Q
                       pad_d, pad_h, pad_w, str_d, str_h, str_w)
         self.updat_kernels = winogradcl.backends.convolution.UpdateCuda(N, C, K, D, H, W, T, R, S, M, P, Q,
                       pad_d, pad_h, pad_w, str_d, str_h, str_w)


