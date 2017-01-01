require "./spec_helper"

def mean(arr)
  arr.reduce(0_f32) { |sum, el| sum + el }.to_f32 / arr.size
end

def std(arr)
  m = mean(arr)
  sum = arr.reduce(0_f32) { |accum, i| accum + (i - m)**2 }
  sample_variance = sum/(arr.size - 1).to_f
  return Math.sqrt(sample_variance)
end

describe MXNet do
  describe "random" do
    it "check_with_device" do
      MXNet::Context.with(MXNet::Context.cpu) do
        a, b = -10_f32, 10_f32
        mu, sigma = 10_f32, 2_f32
        shape = [100, 100]
        MXNet::Random.seed(128)
        ret1 = MXNet::Random.normal(mu, sigma, shape)
        un1 = MXNet::Random.uniform(a, b, shape)
        MXNet::Random.seed(128)
        ret2 = MXNet::Random.normal(mu, sigma, shape)
        un2 = MXNet::Random.uniform(a, b, shape)
        ret1.should eq(ret2)
        un1.should eq(un2)
        (mean(ret1.to_a) - mu).abs.should be < 0.1
        (mean(ret1.to_a) - sigma).abs.should be < 0.1
        (mean(un1.to_a) - (a + b)/2).abs.should be < 0.1
      end
    end
    it "check symbolic random" do
      a, b = -10_f32, 10_f32
      mu, sigma = 10_f32, 2_f32
      shape = [100, 100]
      dev = MXNet::Context.cpu
      _X = MXNet::Symbol.variable("X")
      _Y = MXNet::Symbol.uniform(low: a, high: b, shape: shape) + _X
      x = MXNet::NDArray.zeros(shape, ctx: dev)
      xgrad = MXNet::NDArray.zeros(shape, ctx: dev)
      yexec = _Y.bind(dev, {"X" => x}, {"X" => xgrad})
      MXNet::Random.seed 128
      yexec.forward
      yexec.backward yexec.outputs[0]
      un1 = (yexec.outputs[0] - x).copy_to(dev)
      xgrad.should eq(un1)
      MXNet::Random.seed 128
      yexec.forward
      un2 = (yexec.outputs[0] - x).copy_to(dev)
      un1.should eq(un2)
      (mean(un1.to_a) - (a + b)/2).abs.should be < 0.1
      _Y = MXNet::Symbol.normal(loc: mu, scale: sigma, shape: shape)
      yexec = _Y.bind(dev)
      MXNet::Random.seed 128
      yexec.forward
      ret1 = yexec.outputs[0].copy_to(dev)
      MXNet::Random.seed 128
      ret2 = MXNet::Random.normal(mu, sigma, shape)
      ret1.should eq(ret2)
      (mean(ret1.to_a) - mu).abs.should be < 0.1
      (std(ret1.to_a) - sigma).abs.should be < 0.1
    end
  end
end
