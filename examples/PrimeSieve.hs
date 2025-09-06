module Main where

sieve :: Int -> [Int]
sieve n = sieve' [2..n]
  where
    sieve' [] = []
    sieve' (p:xs) = p : sieve' [ x | x <- xs, x `mod` p /= 0 ]

main :: IO ()
main = print (sieve 200)
