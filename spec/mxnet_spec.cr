require "./spec_helper"

describe MXNet do
  # TODO: Write tests

  it "works" do
    arr = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
    true.should eq(true)
  end
end
