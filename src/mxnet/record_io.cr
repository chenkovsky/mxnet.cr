module MXNet
  class RecordIO
    enum IOFlag
      IOWrite
      IORead
    end
    @record_io_handle : LibMXNet::RecordIOHandle

    def initialize(@uri : String, @flag : IOFlag)
      @record_io_handle = LibMXNet::RecordIOHandle.null
      @is_open = false
      open
    end

    def open
      case @flag
      when IOFlag::IOWrite
        check_call LibMXNet.mx_record_io_writer_create(uri, out @record_io_handle)
      when IOFlag::IORead
        check_call LibMXNet.mx_record_io_reader_create(uri, out @record_io_handle)
      end
      @is_open = true
    end

    def close
      if @is_open
        case flag
        when IOFlag::IOWrite
          check_call LibMXNet.mx_record_io_writer_free(@handle)
        when IOFlag::IORead
          check_call LibMXNet.mx_record_io_reader_free(@handle)
        end
      end
    end

    def reset
      close
      open
    end

    def write(buf : Bytes)
      raise MXError.new "recordio is not writable" unless @flag == IOFlag::IOWrite
      check_call LibMXNet.mx_record_io_writer_write_record(@handle, buf, buf.bytesize)
    end

    def read : Bytes
      raise MXError.new "recordio is not readable" unless @flag == IOFlag::IORead
      result = Pointer(UInt8).null
      size = 0
      check_call LibMXNet.mx_record_io_reader_read_record(@handle, out result, out size)
      Bytes.new result, size
    end

    def pack(header : IRHeader, s : Bytes) : Bytes
      io = IO::Memory.new
      IO::ByteFormat::BigEndian.encode(label.size, io)
      header.label.each do |l|
        IO::ByteFormat::BigEndian.encode(l, io)
      end
      IO::ByteFormat::BigEndian.encode(header.id, io)
      IO::ByteFormat::BigEndian.encode(header.id2, io)
      io.write_utf8 s
      io.flush
      io.close
      io.to_slice
    end

    def unpack(s : Bytes) : {IRHeader, Bytes}
      io = IO::Memory.new s, writable: false
      flag = IO::ByteFormat::BigEndian.decode(Int32, io)
      label = (0...flag).map { |idx| IO::ByteFormat::BigEndian.decode(Float32, io) }
      id = IO::ByteFormat::BigEndian.decode(Int32, io)
      id2 = IO::ByteFormat::BigEndian.decode(Int32, io)
      str = io.read_utf8
      io.close
      return IRHeader.new(label, id, id2), str
    end

    class Indexed < RecordIO
      @idx_path : String

      def initialize(@idx_path, uri, flag)
        super(uri, flag)
        @idx = {} of Int32 => Int32
        if @flag == IOFlag::IORead && File.file? @idx_path
          File.foreach(@idx_path) do |l|
            tmp = line.strip.split "\t"
            k, v = tmp[0].to_i, tmp[1].to_i
            @idx[k] = v
          end
        end
      end

      def close
        if @flag == IOWrite
          File.open(@idx_path, "w") do |f|
            @idx.each do |k, v|
              f.write("#{k}\t#{v}\n")
            end
          end
        end
        super.close
      end

      def reset
        @idx = Hash(Int32, Int32).new
        super.close
        super.open
      end

      def seek(idx : Int32)
        raise MXError.new "flag should be IORead" unless @flag == IORead
        check_call LibMXNet.mx_record_io_reader_seek(@record_io_handle, @idx[isx])
      end

      def tell
        raise MXError.new "flag should be IOWrite" unless @flag == IOWrite
        pos = 0
        check_call LibMXNet.mx_record_io_writer_tell(@record_io_handle, out pos)
        return pos
      end

      def read_idx(idx)
        seek idx
        read
      end

      def write_idx(idx, buf : Bytes)
        pos = tell
        @idx[idx] = pos
        write buf
      end
    end
  end
end
