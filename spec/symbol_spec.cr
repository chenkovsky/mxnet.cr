require "./spec_helper"

def mlp2
  data = MXNet::Symbol.variable("data")
  out_ = data.fully_connected(name: "fc1", num_hidden: 1000)
  out_ = out_.activation(act_type: MXNet::Symbol::ActType::Relu)
  out_ = out_.fully_connected(name: "fc2", num_hidden: 10)
  return data
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
  end
end
