require "logger"

module MXNet
  @@logger = Logger.new(STDOUT)
  @@logger.progname = "MXNetCrystal"

  def logger
    @@logger
  end

  alias MXUInt = UInt32
  alias MXFloat = Float32
  alias MXFloatP = MXFloat*

  class MXError < Exception
  end

  enum MXType
    Float32_T =  0
    Float64_T =  1
    Float16_T =  2
    UInt8_T   =  3
    Int32_T   =  4
    Other_T   = -1

    def self.from_dtype(dtype)
      if dtype == Float32
        return MXType::Float32_T
      elsif dtype == Float64
        return MXType::Float64_T
      elsif dtype == Int32
        return MXType::Int32_T
      else
        return MXType::Other_T
      end
      # case dtype
      # when Float32
      #   MXType::Float32_T
      # when Float64
      #   MXType::Float64_T
      # when Int32
      #   MXType::Int32_T
      # else
      #   MXType::Other_T
      # end
    end

    def self.dtype_id(dtype)
      from_dtype(dtype).value
    end

    def to_dtype
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
    check_call(LibMXNet.mx_notify_shutdown)
  end

  def self.wait_all
    check_call(LibMXNet.mx_ndarray_wait_all)
  end
end
