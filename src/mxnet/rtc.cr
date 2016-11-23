module MXNet
  class Rtc
    @name : String
    @inputs : Array({String, NDArray})
    @outputs : Array({String, NDArray})
    @kernel : String

    def initialize(@name, @inputs, @outputs, @kernel)
      rct_handle = RtcHandle.null
      input_names = @inputs.keys.map &.to_unsafe
      input_nds = @inputs.values &.handle
      output_names = @outputs.keys &.to_unsafe
      output_nds = @outputs.values &.handle
      check_call LibMXNet.mx_rtc_create(name, @inputs.size, @outputs.size,
        input_names, output_names,
        input_nds, output_nds,
        @kernel, out rct_handle)
      @rtc_handle = rtc_handle
    end

    def push(ins : Array(NDArray), outs : Array(NDArray),
             grid_dims : {Int32, Int32, Int32}, block_dims : {Int32, Int32, Int32})
      check_call LibMXNet.mx_rtc_push(@rtc_handle,
        ins.map &.handle,
        outs.map &.handle,
        grid_dims[0],
        grid_dims[1],
        grid_dims[2],
        block_dims[0],
        block_dims[1],
        block_dims[2])
    end

    def finalize
      check_call LibMXNet.mx_rtc_free(@rct_handle)
    end
  end
end
