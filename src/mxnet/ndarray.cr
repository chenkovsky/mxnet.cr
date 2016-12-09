module MXNet
  class NDArray
    @handle : LibMXNet::NDArrayHandle
    @writable : Bool

    def writable?
      @writable
    end

    protected def initialize(@handle, @writable = true)
    end

    def fill(val)
      set val
    end

    def self.zeros(shape : Shape | Array(Int32), ctx : Context? = nil, dtype = MXRealT)
      arr = empty(shape, ctx, dtype)
      arr.fill 0_f32
      arr
    end

    def self.ones(shape : Shape | Array(Int32), ctx : Context? = nil, dtype = MXRealT)
      arr = empty(shape, ctx, dtype)
      arr.fill 1_f32
      arr
    end

    def self.array(src_arr : Array(Float32), shape : Shape, ctx : Context? = nil)
      arr = empty(shape, ctx)
      arr.fill(src_arr)
      arr
    end

    def self.concat(*arrays, ctx : Context? = nil) : NDArray
      concat(arrays.to_a, ctx)
    end

    # Join a sequence of arrays at axis-0
    def self.concat(arrays : Array(NDArray), ctx : Context? = nil) : NDArray
      raise MXError.new "arrays empty" if arrays.size == 0
      array0 = arrays[0]
      axis0 = array0.shape[0]
      shape = array0.shape[1...array0.shape.size]
      (1...arrays.size).each do |idx|
        arr_i = arrays[idx]
        raise MXError.new "shape mismatch between #{arr_i.shape} and #{shape}" unless shape == arr_i.shape[1...arr_i.shape.size]
        axis0 += arr_i.shape[0]
      end
      output = NDArray.empty(Shape.new([axis0] + shape.shape), ctx)
      axis0 = 0
      arrays.each do |arr|
        outputs[axis0...(axis0 + array.shape[0])] = array
        axis0 += array.shape[0]
      end
      output
    end

    def self.load(fname : String) : Hash(String, NDArray) | Array(NDArray)
      out_size = MXUInt.new 0
      out_name_size = MXUInt.new 0
      handles = Pointer(LibMXNet::NDArrayHandle).null
      names = Pointer(UInt8*).null
      check_call LibMXNet.mx_ndarray_load(fname, out out_size, out handles, out out_name_size, out names)
      if out_name_size == 0
        return (0...out_size).map { |idx| NDArray.new handles[idx] }
      elsif out_name_size == out_size
        return Hash.new((0...out_size).map { |idx| String.new names[i] },
          (0...out_size).map { |idx| NDArray.new handles[idx] })
      else
        raise MXError.new "assert out_name_size == 0 || out_name_size == out_size"
      end
      i
    end

    def self.save(fname : String, data : Hash(String, NDArray) | Array(NDArray))
      keys, handles = if data.is_a? Array(NDArray)
                        {nil, data.map { |x| x.handle }}
                      else
                        {data.keys.map { |x| x.to_unsafe }, data.values.map { |x| x.handle }}
                      end
      check_call LibMXNet.mx_ndarray_save(fname, handles.size, handles, keys)
    end

    def self.deserialize(bytes : Bytes)
      handle = LibMXNet::NDArrayHandle.null
      check_call LibMXNet.mx_ndarray_load_from_raw_bytes(bytes, out handle)
      NDArray.new handle
    end

    def serialize : Bytes
      buf = Pointer(UInt8).null
      size = 0
      check_call LibMXNet.mx_ndarray_save_raw_bytes(@handle, out size, out buf)
      return Bytes.new(buf, size)
    end

    private def sync_copy_from(source : Array(Float32))
      raise MXError.new "array size #{source.size} do not match the size of NDArray #{size}" unless source.size == size
      check_call LibMXNet.mx_ndarray_sync_copy_from_cpu(@handle, source, source.size)
    end

    def [](slice : Range(Int, Int))
      slice_handle = LibMXNet::NDArrayHandle.null
      check_call LibMXNet.mx_ndarray_slice(@handle, slice.begin, slice.end, out slice_handle)
      return NDArray.new slice_handle, writable: @writable
    end

    def [](start : Int, count : Int)
      self[start...(start + count)]
    end

    def [](i : Int)
      self[i, 1]
    end

    def reshape(dims : Array(Int32)) : NDArray
      reshape_handle = LibMXNet::NDArrayHandle.null
      check_call LibMXNet.mx_ndarray_reshape(@handle, dims.size, dims, out reshape_handle)
      NDArray.new reshape_handle, writable: @writable
    end

    def reshape(shape : Shape)
      reshape(shape.shape)
    end

    def wait_to_read
      check_call LibMXNet.mx_ndarray_wait_to_read(@handle)
    end

    def context : Context
      dev_type_id = 0
      dev_id = 0
      check_call LibMXNet.mx_ndarray_get_context(@handle, out dev_type_id, out dev_id)
      Context.new Context::DeviceType.new(dev_type), dev_id
    end

    def self.empty(shape : Shape | Array(Int32), ctx : Context? = nil, dtype = MXRealT)
      ctx_ =
        if ctx.nil?
          Context.default_ctx
        else
          ctx
        end
      shape_ = if shape.is_a? Shape
                 shape
               else
                 Shape.new shape
               end
      return NDArray.new new_alloc_handle(shape_, ctx_, false, dtype)
    end

    private def self.new_empty_handle
      hdl = LibMXNet::NDArrayHandle.null
      check_call(LibMXNet.mx_ndarray_create_none(out hdl))
      return hdl
    end

    private def self.new_alloc_handle(shape : Shape, ctx : Context, delay_alloc : Bool, dtype = MXRealT)
      MXNet.check_call(LibMXNet.mx_ndarray_create_ex(
        shape,
        shape.size,
        ctx.device_type.value,
        ctx.device_id,
        delay_alloc ? 1 : 0,
        MXType.dtype_id(dtype),
        out hdl
      ))
      return hdl
    end

    def finalize
      MXNet.check_call(LibMXNet.mx_ndarray_free(@handle))
    end

    def inspect(io)
      shape_info = shape.map { |x| x.to_s }.join("x")
      io << "<#{self.class} #{shape_info} #{@context}>"
    end

    def to_a
      data = Array(Float32).new(size: size, value: 0)
      check_call LibMXNet.mx_ndarray_sync_copy_to_cpu(@handle, data, size)
    end

    def to_scalar
      raise MXError.new "The current array is not a scalar" unless shape.shape == [1]
      return to_a[0]
    end

    def copy_to(other : NDArray)
      if other.handle == @handle
        @@logger.warn("copy an array to itself, is it intended ?")
        other
      else
        _copy_to(other)
      end
    end

    def copy_to(ctx : Context) : NDArray
      ret = NDArray.new shape, ctx, delay_alloc: true
      copy_to(ret)
    end

    def copy
      copy_to(@context)
    end

    def shape
      ndim = MXUInt.new 0
      data = Pointer(MXUInt).null
      check_call LibMXNet.mx_ndarray_get_shape(@handle, out ndim, out data)
      Shape.new (0...ndim).map { |i| data[i] }
    end

    def ==(other : Object)
      case other
      when NDArray
        ohter.shape == shape && other.to_a == to_a
      else
        false
      end
    end
  end
end
