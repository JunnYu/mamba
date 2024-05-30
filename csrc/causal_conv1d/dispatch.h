#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                     \
    if (ITYPE == paddle::DataType::FLOAT16) {                                        \
        using input_t = half;                                                        \
        __VA_ARGS__();                                                               \
    } else if (ITYPE == paddle::DataType::FLOAT32)  {                                \
        using input_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        PADDLE_THROW(#NAME, " not implemented for input type '", ITYPE, "'");        \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == paddle::DataType::FLOAT16) {                                        \
        using weight_t = half;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == paddle::DataType::FLOAT32)  {                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        PADDLE_THROW(#NAME, " not implemented for weight type '", WTYPE, "'");       \
    }