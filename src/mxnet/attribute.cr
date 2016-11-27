module MXNet
  class AttrScope
    # Attribute scoping support for symbolic API.
    getter :attr

    def initialize(@attr : Hash(String, String) = {} of String => String)
    end

    # Get the attribute dict given the attribute set by the symbol.
    # param userDefinedAttr The attribute passed in by user during symbol creation.
    # return Updated attributes to add other scope related attributes.
    def get(attr : Hash(String, String)?)
      return @attr.clone if attr.nil?
      return @attr.merge(attr)
    end

    def self.[](attr)
      @@current.get(attr)
    end

    def self.with(attr_scope)
      old_attr_scope = @@current.clone
      @@current.attr.merge(attr_scope.attr)
      begin
        yield
      ensure
        @@current = old_attr_scope
      end
    end

    @@current = AttrScope.new
  end
end
