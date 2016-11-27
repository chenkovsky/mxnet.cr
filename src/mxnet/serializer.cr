require "base64"

module MXNet
  class Serializer
    include IO::ByteFormat::BigEndian

    private def decode_base64_string(s)
      Base64.decode(s)
    end

    def deserialize_optimizer(s)
      decode_base64_string(s)
    end
  end
end
