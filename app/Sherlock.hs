{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE ViewPatterns        #-}

import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.Primitive
import           Control.Monad.Trans.Class
import           Control.Monad.Trans.State
import           Data.Bifunctor
import           Data.Char
import           Data.Finite
import           Data.Foldable
import           Data.List
import           Data.Maybe
import           Data.Neural.Activation
import           Data.Neural.HMatrix.Recurrent
import           Data.Neural.HMatrix.Recurrent.Dropout
import           Data.Neural.HMatrix.Recurrent.Generate
import           Data.Neural.HMatrix.Recurrent.Train
import           Data.Ord
import           Data.Proxy
import           Data.Tuple
import           GHC.TypeLits
import           GHC.TypeLits.List
import           Numeric.LinearAlgebra.Static
import           Text.Printf
import qualified Data.Map.Strict                        as M
import qualified Data.Set                               as S
import qualified Data.Vector                            as VB
import qualified Data.Vector.Storable                   as V
import qualified Linear.V                               as L
import qualified System.Random.MWC                      as R
import qualified System.Random.MWC.Distributions        as R

genIx :: forall n. KnownNat n => Finite n -> R n
genIx (fromIntegral->i) = fromJust
                        . create
                        $ V.generate l (\i' -> if i' == i then 1 else 0)
  where
    l = fromInteger $ natVal (Proxy @n)

ixMax :: (Foldable t, Ord a) => t a -> Int
ixMax = fst . maximumBy (comparing snd) . zip [0..] . toList

ixSort :: (Foldable t, Ord a) => t a -> [Int]
ixSort = map fst . sortBy (flip (comparing snd)) . zip [0..] . toList

sanitize :: Char -> Char
sanitize c | isPrint c = c
           | otherwise = '#'


main :: IO ()
main = do
    g <- R.create
    holmes <- readFile "data/holmes.txt"
    let holmesVec :: VB.Vector Char
        holmesVec = VB.fromList holmes
    let allCharsV :: VB.Vector Char
        allCharsV = VB.fromList
                  . S.toList
                  . S.fromList
                  $ holmes

    L.reifyVectorNat allCharsV $ \(allChars :: L.V c Char) -> do
      let charMap :: M.Map Char (Finite c)
          charMap = M.fromList
                  . (`zip` [0..])
                  . toList
                  $ allChars
          c2v     :: Char -> R c
          c2v     = genIx . (charMap M.!)
          holmesSeries :: VB.Vector (L.V 15 Char, Char)
          holmesSeries = processSeries $ VB.zip holmesVec (VB.tail holmesVec)
          holmesSeries' :: VB.Vector ([R c], R c)
          holmesSeries' = bimap (map c2v . toList) c2v <$> holmesSeries
          ins :: VB.Vector [R c]
          ins = fst <$> holmesSeries'
          nextChar :: R c -> IO Int
          nextChar i = R.categorical (V.map ((**8) . max 0) (extract i)) g
          fb :: R c -> IO (R c)
          fb = fmap (genIx . fromIntegral) . nextChar
          cb :: Int -> Int -> Network c hs c -> IO ()
          cb e i n = when (i `mod` 1000 == 0) $ do
            let toCharList = map (sanitize . (allCharsV VB.!))
                           . take 50 . ixSort . V.toList . extract
                insTest   = ins VB.! 100
                insChars  = head . toCharList <$> insTest
                (last->lo, pp) = runNetStream naRLLog n (ins VB.! 100)
            lo' <- fb lo
            testOut   <- (lo:) <$> runNetFeedbackM_ naRLLog fb pp 75 lo'
            let testChars = toCharList <$> testOut
            threadDelay 250000
            printf "%d\t%d\n" e i
            mapM_ putStrLn (take 15 testChars)
            testCharsPick <- map (sanitize . (allCharsV VB.!)) <$> mapM nextChar testOut
            putStrLn $ insChars ++ "|" ++ testCharsPick


      print $ M.keys charMap
      print $ natVal (Proxy @c)
      print $ length holmes

      net0 <- randomNetMWC (-0.1,0.1) g :: IO (Network c '[100,75,50] c)
      net1 <- trainHistory 0.2 0.01 0.005 100 holmesSeries' net0 cb g
      return ()


trainHistory
    :: forall i m hs o. (PrimMonad m, KnownNats hs, KnownNat i, KnownNat o)
    => Double       -- ^ dropout
    -> Double       -- ^ step size (weights)
    -> Double       -- ^ step size (state)
    -> Int          -- ^ number of epochs
    -> VB.Vector ([R i], R o)
    -> Network i hs o
    -> (Int -> Int -> Network i hs o -> m ())  -- ^ callback
    -> R.Gen (PrimState m)
    -> m (Network i hs o)
trainHistory d sw ss e samps net0 cb g =
    flip execStateT net0
  . for_ [1 .. e] $ \i -> do
      samps' <- lift $ R.uniformShuffle samps g
      for_ (zip [1..] (VB.toList samps')) $ \(j, (h, t)) ->
        StateT $ \net -> do
          net' <- trainSeriesDOMWC naRLLog d sw ss t h net g
          cb i j net'
          net' `deepseq` return ((), net')

