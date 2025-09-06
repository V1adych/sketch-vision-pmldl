module Main where

-- Simple recursive Sierpinski triangle (ASCII)
tri :: Int -> [String]
tri 0 = ["*"]
tri n = let t = tri (n-1)
            pad = map (\s -> s ++ s) t
        in map (\s -> s ++ s) (map (\s -> ' ' : s) t) ++ pad

main :: IO ()
main = mapM_ putStrLn (tri 5)
