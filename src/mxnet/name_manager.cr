module MXNet
  class NameManager
    @counter : Hash(String, Int32)
    @@current = NameManager.new

    def self.with(nm : NameManager)
      old_manager = @@current
      @@current = nm
      begin
        yield
      ensure
        @current = old_manager
      end
    end

    def initialize
      @count = {} of String => Int32
    end

    def get(name : String?, hint : String) : String
      if name.nil?
        if !@counter.has_key? hint
          @count[hint] = 0
        end
        generated_name = "#{hint}#{@count[hint]}"
        @count[hint] += 1
        generated_name
      else
        name
      end
    end
  end
end
