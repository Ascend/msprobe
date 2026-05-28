# config.json Examples

The following examples contain all configurable parameters in all supported scenarios.

## PyTorch

### task = statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,
    "extra_info": true,

    "statistics": {
        "scope": [], 
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task = tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "bench_path": "/home/bench_data_dump",
        "summary_mode": "md5",
        "diff_nums": 5        
    }
}
```

### task = acc_check

```json
{
    "task": "acc_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "acc_check": {
        "white_list": [],
        "black_list": [],
        "error_data_path": "./"
    }
}
```

### task = structure

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

### Example of Dynamic Control of dump_enable

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "dump_enable": false,
    "statistics": {
        "summary_mode": "statistics"
    }
}
```

> Note: `dump_enable` is configured only when dump needs to be dynamically enabled or disabled. During execution, you can change the value of `dump_enable` from `false` to `true` (or vice versa) to dynamically enable or disable dump. Modifications to other fields in the JSON file also take effect.

## MindSpore Static Graph

### task = statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "statistics": {
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task = tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "tensor": {
        "list":[],
        "data_mode": ["all"]
    }
}
```

### task = overflow_check

```json
{
    "task": "overflow_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "overflow_check": {
        "check_mode": "all"
    }
}
```

### task = exception_dump

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```

## MindSpore Dynamic Graph

### task = statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "statistics": {
        "scope": [], 
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task = tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"]
    }
}
```

### task = structure

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

### task = exception_dump

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```
