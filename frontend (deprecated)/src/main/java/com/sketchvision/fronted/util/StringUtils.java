package com.sketchvision.fronted.util;

import java.util.Collection;
import java.util.StringJoiner;

public final class StringUtils {
    private StringUtils() {}

    public static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }

    public static String repeat(String s, int n) {
        if (s == null || n <= 0) return "";
        StringBuilder sb = new StringBuilder(s.length() * n);
        for (int i = 0; i < n; i++) sb.append(s);
        return sb.toString();
    }

    public static String join(Collection<?> items, String sep) {
        StringJoiner j = new StringJoiner(sep);
        for (Object o : items) j.add(String.valueOf(o));
        return j.toString();
    }
}
