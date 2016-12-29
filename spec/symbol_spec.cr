require "./spec_helper"

describe MXNet do
  describe "symbol" do
    it "variable" do
      data = MXNet::Symbol.variable("data")
      act = data.activation
      # net1 = MXNet::Symbol.fully_connected(name: "fc1", data: data, num_hidden: 10)
    end
  end
end
