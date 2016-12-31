require "file_utils.cr"
require "./spec_helper"

def mlp2
  data = MXNet::Symbol.variable("data")
  out_ = data.fully_connected(name: "fc1", num_hidden: 1000)
  out_ = out_.activation(act_type: MXNet::Symbol::ActType::Relu)
  out_ = out_.fully_connected(name: "fc2", num_hidden: 10)
  return data
end

def conv
  data = MXNet::Symbol.variable("data")
  conv1 = data.convolution(name: "conv1", num_filter: 32, kernel: [3, 3], stride: [2, 2])
  bn1 = conv1.batch_norm(name: "bn1")
  act1 = bn1.activation(name: "relu1", act_type: MXNet::Symbol::ActType::Relu)
  mp1 = act1.pooling(name: "mp1", kernel: [2, 2], stride: [2, 2], pool_type: MXNet::Symbol::PoolType::Max)
  conv2 = mp1.convolution(name: "conv2", num_filter: 32, kernel: [3, 3], stride: [2, 2])
  bn2 = conv2.batch_norm(name: "bn2")
  act2 = bn2.activation(name: "relu2", act_type: MXNet::Symbol::ActType::Relu)
  mp2 = act2.pooling(name: "mp2", kernel: [2, 2], stride: [2, 2], pool_type: MXNet::Symbol::PoolType::Max)

  fl = mp2.flatten(name: "flatten")
  fc2 = fl.fully_connected(name: "fc2", num_hidden: 10)
  softmax = fc2.softmax_output(name: "sm")
  return softmax
end

describe MXNet do
  describe "symbol" do
    it "mlp2" do
      m = mlp2
      m.arguments
      m.outputs
    end

    it "compose" do
      data = MXNet::Symbol.variable("data")
      net1 = data.fully_connected(name: "fc1", num_hidden: 10)
      net1 = net1.fully_connected(name: "fc2", num_hidden: 100)
      net1.arguments.should eq(["data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"])

      net2 = MXNet::Symbol.fully_connected(name: "fc3", num_hidden: 10)
      net2 = net2.activation(act_type: MXNet::Symbol::ActType::Relu)
      net2 = net2.fully_connected(name: "fc4", num_hidden: 20)
      composed = net2.call(fc3_data: net1, name: "composed")
      multi_out = MXNet::Symbol.group [composed, net1]
      multi_out.outputs.size.should eq(2)
    end
    it "copy" do
      data = MXNet::Symbol.variable("data")
      data2 = data.clone
      data3 = data.dup
      data.to_json.should eq(data2.to_json)
      data.to_json.should eq(data3.to_json)
    end

    it "internal" do
      data = MXNet::Symbol.variable("data")
      oldfc = data.fully_connected(name: "fc1", num_hidden: 10)
      net1 = oldfc.fully_connected(name: "fc2", num_hidden: 100)
      net1.arguments.should eq(["data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"])
      internal = net1.internals
      fc1 = internal["fc1_output"]
      fc1.arguments.should eq(oldfc.arguments)
    end

    it "from json" do
      m0 = mlp2
      MXNet::Symbol.from_json(m0.to_json).to_json.should eq(m0.to_json)
      m1 = conv
      MXNet::Symbol.from_json(m1.to_json).to_json.should eq(m1.to_json)
    end

    it "save load" do
      sym = mlp2
      fname = "tmp_sym.json"
      sym.save(fname)
      data2 = MXNet::Symbol.load fname
      sym.to_json.should eq(data2.to_json)
      FileUtils.rm fname
    end

    it "infer type" do
      data = MXNet::Symbol.variable("data")
      f32data = data.cast MXNet::MXType::Float32_T
      fc1 = f32data.fully_connected(name: "fc1", num_hidden: 128)
      mlp = fc1.softmax_output(name: "softmax")

      arg, out_, aux = mlp.infer_type(data: MXNet::MXType::Float16_T)
      arg.should eq([MXNet::MXType::Float16_T, MXNet::MXType::Float32_T, MXNet::MXType::Float32_T, MXNet::MXType::Float32_T])
      out_.should eq([MXNet::MXType::Float32_T])
      aux.should eq([] of MXNet::MXType)
    end
    it "infer shape" do
      num_hidden = 128
      num_dim = 64
      num_sample = 10
      data = MXNet::Symbol.variable("data")
      prev = MXNet::Symbol.variable("prevstate")
      x2h = data.fully_connected(name: "x2h", num_hidden: num_hidden)
      h2h = prev.fully_connected(name: "h2h", num_hidden: num_hidden)
      out_ = (x2h + h2h).activation(name: "out", act_type: MXNet::Symbol::ActType::Relu)
      ret = out_.infer_shape(data: [num_sample, num_dim])
      ret.should eq([nil, nil, nil])
      arg, out_shapes, aux_shapes = out_.infer_shape_partial(data: [num_sample, num_dim])
      arg_shapes = out_.arguments.zip(arg).to_h
      arg_shapes["data"].should eq([num_sample, num_dim])
      arg_shapes["x2h_weight"].should eq([num_hidden, num_dim])
      arg_shapes["h2h_weight"].size.should eq(0)

      state_shape = out_shapes[0]
      arg, out_shapes, aux_shapes = out_.infer_shape(data: [num_sample, num_dim], prevstate: state_shape)
      arg_shapes = out_.arguments.zip(arg).to_h
      arg_shapes["data"].should.eq([num_sample, num_dim])
      arg_shapes["x2h_weight"].should eq([num_hidden, num_dim])
      arg_shapes["h2h_weight"].should eq([num_hidden, num_hidden])
    end
    it "infer shape var" do
      shape = [2, 3]
      a = MXNet::Symbol.variable("a", shape: shape)
      b = MXNet::Symbol.variable("b")
      c = a + b
      arg_shapes, out_shapes, aux_shapes = c.infer_shape
      arg_shapes[0].should eq(shape)
      arg_shapes[1].should eq(shape)
      arg_shapes[0].should eq(shape)

      overwrite_shape = [5, 6]
      arg_shapes, out_shapes, aux_shapes = c.infer_shape a: overwrite_shape
      arg_shapes[0].should eq(overwrite_shape)
      arg_shapes[1].should eq(overwrite_shape)
      out_shapes[0].should eq(overwrite_shape)
    end
  end
end
