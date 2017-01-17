require "./spec_helper"
describe MXNet do
  describe "attr" do
    it "attr_basic" do
      MXNet::AttrScope.with(MXNet::AttrScope.new({"group" => "4",
        "data"  => "great"})) do
        data = MXNet::Symbol.variable("data", attr: {
          "dtype" => "data", "group" => "1", "force_mirroring" => "True",
          "lr_mult" => "1",
        })
        gdata = MXNet::Symbol.variable("data2")
        gdata.attr("group").should eq("4")
        data.attr("group").should eq("1")
        data.attr("lr_mult").should eq("1")
        data.attr("__lr_mult__").should eq("1")
        data.attr("force_mirroring").should eq("True")
        data.attr("__force_mirroring__").should eq("True")
        data2 = MXNet::Symbol.from_json data.to_json
        data.attr("dtype").should eq(data2.attr("dtype"))
      end
    end
    it "operator" do
      data = MXNet::Symbol.variable("data")
      MXNet::AttrScope.with(MXNet::AttrScope.new({"__group__" => "4",
        "__data__"  => "great"})) do
        fc1 = data.activation(act_type: MXNet::Symbol::ActType::Relu)
        MXNet::AttrScope.with(MXNet::AttrScope.new({"__init_bias__" => "0.0"})) do
          fc2 = fc1.fully_connected(num_hidden: 10, name: "fc2")
          fc1.attr("__data__").should eq("great")
          fc2.attr("__data__").should eq("great")
          fc2.attr("__init_bias__").should eq("0.0")
          fc2copy = MXNet::Symbol.from_json fc2.to_json
          fc2copy.to_json.should eq(fc2.to_json)
          fc2weight = fc2.internals["fc2_weight"]
        end
      end
    end

    it "list attr" do
      data = MXNet::Symbol.variable("data", attr: {"mood" => "angry"})
      op = data.convolution(name: "conv", kernel: [1, 1],
        num_filter: 1, attr: {"__mood__" => "so so", "wd_mult" => "x"})
      op.attr("__mood__").should eq("so so")
      op.attr("wd_mult").should eq("x")
      op.attr("__wd_mult__").should eq("x")
    end
    # it "attr dict" do
    #   data = MXNet::Symbol.variable("data", attr: {"mood" => "angry"})
    #   op = data.convolution(name: "conv", kernel: [1, 1],
    #     num_filter: 1, attr: {"__mood__" => "so so", "lr_mult" => "1"})
    #   dict = op.attr_dict
    #   dict["data"]["mood"]?.should eq("angry")
    #   dict["conv_weight"]["__mood__"]?.should eq("so so")
    #   dict["conv"]["kernel"]?.should eq("(1, 1)")
    #   dict["conv"]["__mood__"]?.should eq("so so")
    #   dict["conv"]["num_filter"]?.should eq("1")
    #   dict["conv"]["lr_mult"]?.should eq("1")
    #   dict["conv"]["__lr_mult__"]?.should eq("1")
    #   dict["conv_bias"]["__mood__"]?.should eq("so so")
    # end
  end
end
