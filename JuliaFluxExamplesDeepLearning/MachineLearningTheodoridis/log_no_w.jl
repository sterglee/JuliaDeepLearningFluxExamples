"""
log_no_w(x)

Returns the natural logarithm of x.
In Julia, log(0) returns -Inf without a warning, so no suppression is required.
"""
function log_no_w(x)
    return log.(x)
end
