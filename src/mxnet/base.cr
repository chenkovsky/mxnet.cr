require "logger"

module MXNet
  @@logger = Logger.new(STDOUT)
  @@logger.progname = "MXNetCrystal"

  def self.logger
    @@logger
  end

  alias MXUInt = UInt32
  alias MXFloat = Float32
  alias MXScalar = Float32 | Float64 | UInt8 | Int32
  alias MXArray = Array(Float32) | Array(Float64) | Array(UInt8) | Array(Int32)

  class MXError < Exception
  end

  enum MXType
    Float32_T =  0
    Float64_T =  1
    Float16_T =  2
    UInt8_T   =  3
    Int32_T   =  4
    Other_T   = -1

    def array(size : Int32) : MXArray
      case self
      when MXType::Float32_T
        Array(Float32).new(size, 0_f32)
      when MXType::Float64_T
        Array(Float64).new(size, 0_f64)
      when MXType::Float16_T
        Array(Float32).new(size, 0_f32)
      when MXType::UInt8_T
        Array(UInt8).new(size, 0_u8)
      when MXType::Int32_T
        Array(Int32).new(size, 0)
      else
        raise MXError.new "invalid MXType: #{self}"
      end
    end

    private def self.from_dtype(dtype)
      case dtype
      when Float32.class
        MXType::Float32_T
      when Float64.class
        MXType::Float64_T
      when Int32.class
        MXType::Int32_T
      else
        MXType::Other_T
      end
    end

    def self.dtype_id(dtype)
      from_dtype(dtype).value
    end

    private def to_dtype
      case self
      when MXType::Float32_T
        Float32
      when MXType::Float64_T
        Float64
      when MXType::Float16_T
        Float16
      when MXType::UInt8_T
        UInt8
      when MXType::Int32_T
        Int32
      else
        raise MXError.new "Cannot convert MXType: #{type_} to dtype"
      end
    end
  end

  def self.check_call(ret)
    raise MXError.new (String.new LibMXNet.mx_get_last_error) if ret != 0
  end

  private def self.notify_shutdown
    MXNet.check_call(LibMXNet.mx_notify_shutdown)
  end

  def self.wait_all
    MXNet.check_call(LibMXNet.mx_ndarray_wait_all)
  end
end
