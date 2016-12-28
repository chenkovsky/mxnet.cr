module MXNet
  class NDArray
    def to_unsafe
      @handle
    end

    @handle : LibMXNet::NDArrayHandle
    @writable : Bool

    def writable?
      @writable
    end

    protected def initialize(@handle, @writable = true)
    end

    def fill(val : MXFloat | NDArray | Array(MXFloat))
      set val
    end

    def self.zeros(shape : Shape | Array(Int32), ctx : Context? = nil, dtype = MXFloat)
      arr = empty(shape, ctx, dtype)
      arr.fill 0_f32
      arr
    end

    def self.ones(shape : Shape | Array(Int32), ctx : Context? = nil, dtype = MXFloat)
      arr = empty(shape, ctx, dtype)
      arr.fill 1_f32
      arr
    end

    def self.array(src_arr : Array(Float32), shape : Shape | Array(Int32), ctx : Context? = nil)
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
      output = NDArray.empty([axis0] + shape.shape, ctx)
      axis0 = 0
      arrays.each do |arr|
        output[axis0...(axis0 + arr.shape[0])] = arr
        axis0 += arr.shape[0]
      end
      output
    end

    def self.load(fname : String) : Hash(String, NDArray) | Array(NDArray)
      MXNet.check_call LibMXNet.mx_ndarray_load(fname, out out_size, out handles, out out_name_size, out names)
      if out_name_size == 0
        return (0...out_size).map { |idx| NDArray.new handles[idx] }
      elsif out_name_size == out_size
        ret = Hash(String, NDArray).new initial_capacity: out_size
        (0...out_size).each do |idx|
          ret[String.new names[idx]] = NDArray.new handles[idx]
        end
        return ret
      else
        raise MXError.new "assert out_name_size == 0 || out_name_size == out_size"
      end
    end

    def save(fname : String, name : String? = nil)
      if name
        save fname, {name => self}
      else
        save fname, [self]
      end
    end

    def self.save(fname : String, data : Hash(String, NDArray) | Array(NDArray))
      keys, handles = if data.is_a? Array(NDArray)
                        {nil, data.map { |x| x.to_unsafe }}
                      else
                        {data.keys.map { |x| x.to_unsafe }, data.values.map { |x| x.to_unsafe }}
                      end
      MXNet.check_call LibMXNet.mx_ndarray_save(fname, handles.size, handles, keys)
    end

    def self.deserialize(bytes : Bytes)
      MXNet.check_call LibMXNet.mx_ndarray_load_from_raw_bytes(bytes, bytes.size, out handle)
      NDArray.new handle
    end

    def serialize : Bytes
      MXNet.check_call LibMXNet.mx_ndarray_save_raw_bytes(@handle, out size, out buf)
      return Bytes.new(buf, size)
    end

    def [](slice : Range(MXUInt, MXUInt) | Range(Int32, Int32))
      end_ = slice.end + (slice.exclusive? ? 0 : 1)
      MXNet.check_call LibMXNet.mx_ndarray_slice(@handle, slice.begin, end_, out slice_handle)
      return NDArray.new slice_handle, writable: @writable
    end

    def [](start : MXUInt | Int32, count : MXUInt | Int32)
      self[start...(start + count)]
    end

    def [](i : MXUInt | Int32)
      self[i, 1]
    end

    def []=(slice, val)
      arr = self[slice]
      arr.fill val
      self
    end

    def []=(start, count, val)
      arr = self[(start...(start + count))]
      arr.fill val
      self
    end

    def reshape(dims : Array(Int32)) : NDArray
      MXNet.check_call LibMXNet.mx_ndarray_reshape(@handle, dims.size, dims, out reshape_handle)
      NDArray.new reshape_handle, writable: @writable
    end

    def reshape(shape : Shape)
      reshape(shape.shape)
    end

    def wait_to_read
      MXNet.check_call LibMXNet.mx_ndarray_wait_to_read(@handle)
    end

    def context : Context
      MXNet.check_call LibMXNet.mx_ndarray_get_context(@handle, out dev_type_id, out dev_id)
      Context.new Context::DeviceType.new(dev_type), dev_id
    end

    def self.empty(shape : Shape | Array(Int32) | Array(UInt32) | Nil = nil, ctx : Context? = nil, dtype = MXFloat)
      if shape.nil?
        MXNet.check_call LibMXNet.mx_ndarray_create_none(out hdl)
        return NDArray.new hdl
      end
      ctx_ = ctx.nil? ? Context.default_ctx : ctx
      shape_ = shape.is_a?(Shape) ? shape : Shape.new(shape.map { |x| x.to_u32 })
      return NDArray.new new_alloc_handle(shape_, ctx_, false, dtype)
    end

    private def self.new_alloc_handle(shape : Shape, ctx : Context, delay_alloc : Bool, dtype = MXFloat)
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

    def size
      shape.product
    end

    def to_a
      size_ = size
      data = Array(Float32).new(size_, 0_f32)
      MXNet.check_call LibMXNet.mx_ndarray_sync_copy_to_cpu(@handle, data, size_)
      return data
    end

    def to_scalar
      raise MXError.new "The current array is not a scalar" unless shape.shape == [1]
      return to_a[0]
    end

    def copy_to(other : NDArray)
      if other.to_unsafe == @handle
        MXNet.logger.warn("copy an array to itself, is it intended ?")
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
      MXNet.check_call LibMXNet.mx_ndarray_get_shape(@handle, out ndim, out data)
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

    def dtype
      MXNet.check_call LibMXNet.mx_ndarray_get_dtype(@handle, out dtype_id)
      to_dtype MXType.new dtype_id
    end
  end
end
