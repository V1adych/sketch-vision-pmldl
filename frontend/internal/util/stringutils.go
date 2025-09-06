package util

import (
	"fmt"
	"strings"
)

func IsBlank(s string) bool {
	return len(strings.TrimSpace(s)) == 0
}

func Repeat(s string, n int) string {
	if n <= 0 || len(s) == 0 {
		return ""
	}
	var b strings.Builder
	b.Grow(len(s) * n)
	for i := 0; i < n; i++ {
		b.WriteString(s)
	}
	return b.String()
}

func Join(items []any, sep string) string {
	var b strings.Builder
	for i, it := range items {
		if i > 0 {
			b.WriteString(sep)
		}
		b.WriteString(toString(it))
	}
	return b.String()
}

func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	default:
		return strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(fmtSprint(v)), "\n", " "), "\t", " "))
	}
}

func fmtSprint(v any) string { return fmtSprintf("%v", v) }
func fmtSprintf(format string, a ...any) string { return sprintf(format, a...) }

// lightweight indirection to avoid importing fmt in the API surface
func sprintf(format string, a ...any) string { return _sprintf(format, a...) }
var _sprintf = func(format string, a ...any) string { return fmt.Sprintf(format, a...) }
