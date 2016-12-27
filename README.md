# mxnet
Under developping
mxnet bind for crystal-lang


## Installation

install [MXNet](https://github.com/dmlc/mxnet)

```bash
export LIBRARY_PATH=/path_to_mxnet/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/path_to_mxnet/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/path_to_mxnet/lib:$DYLD_LIBRARY_PATH
```

Add this to your application's `shard.yml`:

```yaml
dependencies:
  mxnet:
    github: chenkovsky/mxnet.cr
```

## Usage


```crystal
require "mxnet"
```


TODO: Write usage instructions here

## Development

TODO: Write development instructions here

## Contributing

1. Fork it ( https://github.com/chenkovsky/mxnet/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [chenkovsky](https://github.com/chenkovsky) chenkovsky - creator, maintainer
