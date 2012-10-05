require 'torch'
require 'dok'
require 'image'
require 'xlua'

local help_desc = [[
      OpenCV 2.4 wrapper
]]

opencv24 = {}

-- load C lib
require 'libopencv24'

--------------------------------------------------------------------------------
-- Image conversions
--

function opencv24.TH2CVImage(im)
   if (im:type() == 'torch.ByteTensor') and ((im:nDimension() == 2) or (im:size(3) == 3)) then
      -- TODO: in the unlikely case of a 3-channels 3xh byte image, this fails
      return im
   else
      local im_cv = torch.ByteTensor()
      im.libopencv24.TH2CVImage(im, im_cv)
      return im_cv
   end
end

function opencv24.CV2THImage(im_cv)
   local im = torch.Tensor()
   im.libopencv24.CV2THImage(im_cv, im)
   return im
end

--------------------------------------------------------------------------------
-- Tracking
--

function opencv24.TrackPointsLK(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'opencv24.TrackPointsLK', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
      {arg='maxPoints', type='number', help='Maximum number of tracked points', default=500},
      {arg='pointsQuality',type='number',help='Minimum quality of trackedpoints',default=0.02},
      {arg='pointsMinDistance', type='number',
       help='Minumum distance between two tracked points', default=3.0},
      {arg='featuresBlockSize', type='number',
       help='opencv GoodFeaturesToTrack block size', default=20},
      {arg='trackerWinSize', type='number',
       help='opencv calcOpticalFlowPyrLK block size', default=11},
      {arg='trackerMaxLevel', type='number',
       help='opencv GoodFeaturesToTrack pyramid depth', default=5},
      {arg='useHarris', type='bool', default = false, help = 'Use Harris detector'})

   if self.im1:size(1) == 3 then
      self.im1 = opencv24.TH2CVImage(self.im1)
   end
   if self.im2:size(1) == 3 then
      self.im2 = opencv24.TH2CVImage(self.im2)
   end

   local corresps = torch.FloatTensor(self.maxPoints, 4)
   libopencv24.TrackPoints(self.im1, self.im2, corresps, self.maxPoints, self.pointsQuality,
			   self.pointsMinDistance, self.featuresBlockSize,
			   self.trackerWinSize, self.useHarris)
   return corresps
end

function opencv24.TrackPointsFREAK(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'opencv24.TrackPointsFREAK', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
      {arg='im1Freaks', type='torch.ByteTensor', default=nil,
       help='Precomputed FREAKS in image 1 (replaces im1)'},
      {arg='im2Freaks', type='torch.ByteTensor', default=nil,
       help='Precomputed FREAKS in image 2 (replaces im2)'},
      {arg='iFREAK', type='number', default = nil,
       help='FREAK object index (cf. opencv24.CreateFREAK)'},
      {arg='detectionThres', type='number', default=40,
       help='FAST detector threshold'},
      {arg='matchingThres', type='number', default=100,
       help='FREAK matching threshold (Hamming distance)'})
   
   if not self.im1Freaks then
      self.iFREAK = self.iFREAK or opencv24.CreateFREAK()
      self.im1Freaks = opencv24.ComputeFREAK(self.im1, self.detectionThres, self.iFREAK)
   end
   if not self.im2Freaks then
      self.iFREAK = self.iFREAK or opencv24.CreateFREAK()
      self.im2Freaks = opencv24.ComputeFREAK(self.im2, self.detectionThres, self.iFREAK)
   end
   local matches = opencv24.MatchFREAK(self.im1Freaks, self.im2Freaks, self.matchingThres)   
   local tracked = torch.Tensor(matches:size(1), 4)
   for i = 1,matches:size(1) do
      tracked[{i, {1,2}}]:copy(self.im1Freaks.pos[{matches[i][1], {1,2}}])
      tracked[{i, {3,4}}]:copy(self.im2Freaks.pos[{matches[i][2], {1,2}}])
   end
   return tracked
end

--------------------------------------------------------------------------------
-- Dense Optical Flow
--

function opencv24.DenseOpticalFlow(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'opencv24.DenseOpticalFlow', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
      {arg='pyr_scale', type='number', default=0.5,
       help='Ratio between 2 successive pyramid scales'},
      {arg='levels', type='number', default=5, help='Pyramid depth'},
      {arg='winsize', type='number', default=11, 
       help='Averaging window size'},
      {arg='iterations', type='number', default=20, 
       help='Number of iteration at each level'},
      {arg='poly_n', type='number', default=5,
       help='Size of the pixel neighborhood used to find polynomial expansion in each pixel'},
      {arg='poly_sigma', type='number', default=1.1,
       help='Size of the pixel neighborhood used to find polynomial expansion in each pixel. For poly_n=5 , you can set poly_sigma=1.1 . For poly_n=7 , a good value would be poly_sigma=1.5'})
   local flow = torch.Tensor(self.im1:size(2), self.im1:size(3), 2)
   local im1_cv = opencv24.TH2CVImage(self.im1)
   local im2_cv = opencv24.TH2CVImage(self.im2)
   flow.libopencv24.DenseOpticalFlow(im1_cv, im2_cv, flow, 
                                     self.pyr_scale, self.levels,
				     self.winsize, self.iterations, 
                                     self.poly_n, self.poly_sigma)
   return flow
end

--------------------------------------------------------------------------------
-- CornerHarris
--

function opencv24.CornerHarris(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'opencv24.CornerHarris', help_desc,
      {arg='im', type='torch.Tensor', help='image'},
      {arg='blocksize', type='number', default=9, 
       help='Neighborhood size (See. opencv  cornerEigenValsAndVecs())'},
      {arg='ksize', type='number', default=3, 
       help='Aperture parameter for the Sobel() operator.'},
      {arg='k', type='number', default=0.04, 
       help='Harris detector free parameter.'})
   local out = torch.Tensor(self.im:size(2), self.im:size(3))
   local im_cv = opencv24.TH2CVImage(self.im)
   out.libopencv24.CornerHarris(im_cv, out, 
                                self.blocksize,self.ksize,self.k)
   return out
end

--------------------------------------------------------------------------------
-- CornerHarris
--

function opencv24.DetectExtract(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'opencv24.DetectExtract', help_desc,
      {arg='im', type='torch.Tensor', help='image'},
      {arg='mask', type='torch.Tensor',
       help='mask areas where not to compute.', 
       default=torch.Tensor()},
      {arg='detectorType', type="string",
       help="GFTT etc.",default="FAST"},
      {arg='extractorType', type="string",
       help="FREAK etc.",default="SURF"},
      {arg='maxPoints', type='number', 
       help='Maximum number of tracked points', default=0},
      {arg='pointsQuality',type='number',
       help='Minimum quality of trackedpoints',default=0.02},
      {arg='pointsMinDistance', type='number',
       help='Minumum distance between two tracked points', default=3.0},
      {arg='blocksize', type='number', default=9, 
       help='Neighborhood size (See. opencv  cornerEigenValsAndVecs())'},
      {arg='useHarris', type='bool', default = false, 
       help = 'Use Harris detector'},
      {arg='k', type='number', default=0.04, 
       help='Harris detector free parameter.'})
   local positions = torch.Tensor(self.maxPoints, 2)
   local feat      = torch.Tensor(self.maxPoints, 128)
   local im_cv     = opencv24.TH2CVImage(self.im)
   feat.libopencv24.DetectExtract(im_cv, self.mask, positions, feat, 
                                  self.detectorType, self.extractorType,
                                  self.maxPoints)
   return positions,feat
end

--------------------------------------------------------------------------------
-- FREAK
--

function opencv24.CreateFREAK(orientedNormalization, scaleNormalization,
			      patternSize, nOctave, trainedPairs)
   orientedNormalization = orientedNormalization or true
   scaleNormalization = scaleNormalization or true
   patternSize = patternSize or 22
   nOctave = nOctave or 4
   trainedPairs = trainedPairs or torch.IntTensor()
   local iFREAK = libopencv24.CreateFREAK(orientedNormalization, scaleNormalization,
					  patternSize, nOctave, trainedPairs)
   opencv24.ComputeFREAK(image.lena(), 40, iFREAK)
   return iFREAK
end

function opencv24.DeleteFREAK(iFREAK)
   libopencv24.DeleteFREAK(iFREAK)
end

function opencv24.ComputeFREAKfromKeyPoints(im, kp, iFREAK)
   local freaks = {}
   freaks.descs = torch.ByteTensor()
   freaks.pos   = kp
   libopencv24.ComputeFREAKfromKeyPoints(opencv24.TH2CVImage(im), 
                                         freaks.descs, freaks.pos,
                                         detection_threshold, iFREAK);
   return freaks
end

function opencv24.ComputeFREAK(im, detection_threshold, iFREAK)
   local freaks = {}
   freaks.descs = torch.ByteTensor()
   freaks.pos = torch.FloatTensor()
   libopencv24.ComputeFREAK(opencv24.TH2CVImage(im), freaks.descs, freaks.pos,
			    detection_threshold, iFREAK);
   return freaks
end

function opencv24.DrawFREAK(im, freaks, r, g, b)
   r = r or 1
   g = g or 0
   b = b or 0
   require 'draw'
   for i = 1,freaks.pos:size(1) do
      local x = freaks.pos[i][1]
      local y = freaks.pos[i][2]
      local rad = freaks.pos[i][3]
      local ang = freaks.pos[i][4]
      draw.circle(im, x, y, rad, r, g, b)
      draw.line(im, x, y, x+rad*math.cos(ang), y+rad*math.sin(ang), r, g, b)
   end
end

function opencv24.MatchFREAK(freaks1, freaks2, threshold)
   local matches = torch.LongTensor()
   local nMatches = libopencv24.MatchFREAK(freaks1.descs, freaks2.descs, matches, threshold)
   if nMatches == 0 then
      return torch.Tensor()
   else
      matches:add(1) -- one-based lua
      return matches:narrow(1,1,nMatches)
   end
end

function opencv24.TrainFREAK(images, iFREAK, keypoints_threshold, correlation_threshold)
   if #images < 1 then
      error("opencv24.TrainFREAK : there must be at least one image...")
   end
   local images_cv = {}
   for i = 1,#images do
      images_cv[i] = opencv24.TH2CVImage(images[i])
   end
   local pairs = torch.IntTensor()
   libopencv24.TrainFREAK(images_cv, pairs, iFREAK, keypoints_threshold,
			  correlation_threshold)
   return pairs
end

function opencv24.ComputeFAST(im, detection_threshold)
   local pos = torch.FloatTensor()
   libopencv24.ComputeFAST(opencv24.TH2CVImage(im), pos, 
                           detection_threshold);
   return pos
end

function opencv24.DrawFAST(im, pos, r, g, b)
   r = r or 1
   g = g or 0
   b = b or 0
   require 'draw'
   for i = 1,pos:size(1) do
      local x   = pos[i][1]
      local y   = pos[i][2]
      local rad = pos[i][3]
      local ang = pos[i][4]
      draw.circle(im, x, y, rad, r, g, b)
      draw.line(im, x, y, 
                x+rad*math.cos(ang), y+rad*math.sin(ang), 
                r, g, b)
   end
end

function opencv24.DrawPos(im, pos, size, r, g, b)
   r = r or 1
   g = g or 0
   b = b or 0
   require 'draw'
   for i = 1,pos:size(1) do
      local x   = pos[i][1]
      local y   = pos[i][2]
      draw.circle(im, x, y, size , r, g, b)
   end
end

function opencv24.Version()
   libopencv24.Version()
end

--------------------------------------------------------------------------------
-- Test/Example
--

function opencv24.ImageConversion_testme()
   local im = image.scale(image.lena(), 123, 242)
   local eps = 1/(255*2)
   local im_cv = opencv24.TH2CVImage(im)
   local im2 = opencv24.CV2THImage(im_cv)
   local diff = (im-im2):abs():gt(eps):sum()
   assert(diff == 0)
   local im3 = opencv24.CV2THImage(opencv24.TH2CVImage(im_cv))
   diff = (im-im3):abs():gt(eps):sum()
   assert(diff == 0)

   im = im[1]
   im_cv = opencv24.TH2CVImage(im)
   im2 = opencv24.CV2THImage(im_cv)
   diff = (im-im2):abs():gt(eps):sum()
   assert(diff == 0)
   im3 = opencv24.CV2THImage(opencv24.TH2CVImage(im_cv))
   diff = (im-im3):abs():gt(eps):sum()
   assert(diff == 0)
end

function opencv24.TrackPointsLK_testme()
   require 'draw'
   local im = image.lena()
   local im2 = image.rotate(im, 0.1)
   local timer = torch.Timer()
   local corresps = opencv24.TrackPointsLK{im1=im, im2=im2, maxPoints = 100}
   print("Track : ", timer:time().real)
   local disp = torch.Tensor(3, im:size(2), im:size(3)*2)
   disp[{{},{},{1,im:size(3)}}]:copy(im)
   disp[{{},{},{im:size(3)+1,im:size(3)*2}}]:copy(im2)
   for i = 1,corresps:size(1) do
      local x1 = corresps[i][1]
      local y1 = corresps[i][2]
      local x2 = corresps[i][3]
      local y2 = corresps[i][4]
      draw.line(disp, x1, y1, x2+im:size(3), y2, 0, 0, 1)
   end
   image.display{image=disp, zoom=1}
end

function opencv24.TrackPointsFREAK_testme()
   require 'draw'
   local im = image.lena()
   local im2 = image.rotate(im, 0.1)
   local corresps = opencv24.TrackPointsFREAK{im1=im, im2=im2}
   local disp = torch.Tensor(3, im:size(2), im:size(3)*2)
   disp[{{},{},{1,im:size(3)}}]:copy(im)
   disp[{{},{},{im:size(3)+1,im:size(3)*2}}]:copy(im2)
   for i = 1,corresps:size(1) do
      local x1 = corresps[i][1]
      local y1 = corresps[i][2]
      local x2 = corresps[i][3]
      local y2 = corresps[i][4]
      draw.line(disp, x1, y1, x2+im:size(3), y2, 0, 0, 1)
   end
   image.display{image=disp, zoom=1}
end

function opencv24.FREAK_testme()
   local size = 22
   local iFREAK = opencv24.CreateFREAK(true, true, size, 4)
   local im = image.lena()
   local im2 = image.rotate(im, 0.1)
   --local im  = image.load('/home/myrhev/NYU/depth-estimation/radial/data/no-risk/part1/images/000000001.jpg')
   --local im2 = image.load('/home/myrhev/NYU/depth-estimation/radial/data/no-risk/part1/images/000000002.jpg')
   local trainedPairs = opencv24.TrainFREAK({im}, iFREAK, 40, 0.7)
   local iFREAK = opencv24.CreateFREAK(true, true, size, 4, trainedPairs)
   local timer = torch.Timer()
   local freaks = opencv24.ComputeFREAK(im, 40, iFREAK)
   print("Freak 1 : ", timer:time().real)
   local freaks2 = opencv24.ComputeFREAK(im2, 40, iFREAK)
   print("Freak 2 : ", timer:time().real)
   local matches = opencv24.MatchFREAK(freaks, freaks2, 100)
   print("Matches : ", timer:time().real)
   local imb = im:clone()
   local im2b = im2:clone()
   opencv24.DrawFREAK(imb, freaks)
   opencv24.DrawFREAK(im2b, freaks2)
   local disp = torch.Tensor(3, imb:size(2), imb:size(3)*2)
   disp[{{},{},{1,imb:size(3)}}]:copy(imb)
   disp[{{},{},{imb:size(3)+1,imb:size(3)*2}}]:copy(im2b)
   for i = 1,matches:size(1) do
      local x1 = freaks.pos[matches[i][1]][1]
      local y1 = freaks.pos[matches[i][1]][2]
      local x2 = freaks2.pos[matches[i][2]][1]
      local y2 = freaks2.pos[matches[i][2]][2]
      draw.line(disp, x1, y1, x2+imb:size(3), y2, 0, 0, 1)
   end
   image.display{image=disp, zoom=1}
   opencv24.DeleteFREAK(iFREAK)
end

function opencv24.FAST_testme()
   local im    = image.lena()
   local timer = torch.Timer()
   local t0    = timer:time().real
   local pos   = opencv24.ComputeFAST(im,40)
   local t1    = timer:time().real
   print("FAST : ", t1-t0)
   opencv24.DrawFAST(im,pos)
   image.display{image=im, zoom=1}
end

function opencv24.CornerHarris_testme()
   local im    = image.lena()
   local timer = torch.Timer()
   local t0    = timer:time().real
   local cmap  = opencv24.CornerHarris(im)
   local t1    = timer:time().real
   print("CornerHarris : ", t1-t0)
   image.display{image=cmap, zoom=1}
   return cmap
end

function opencv24.DetectExtract_testme(dtype,etype)
   if not dtype then
      dtype = "FAST"
   end
   if not etype then 
      etype = "SURF"
   end
   -- require 'draw'
   local im = image.lena()
   local m  = torch.Tensor(im:size(2),im:size(3)):fill(0)
   m:narrow(1,100,200):narrow(2,100,200):fill(1)
   local timer = torch.Timer()
   local pos, feat = 
      opencv24.DetectExtract{im=im[1], maxPoints = 100,
                             mask=m,
                             detectorType=dtype,
                             extractorType=etype}
   local pos2, feat2 = 
      opencv24.DetectExtract{im=im[1], 
                             detectorType=dtype,
                             extractorType=etype}
   print("DetectExtract : ", timer:time().real)
   -- opencv24.DrawPos(im,pos,11)
   im[1]:cmul(m)
   for i = 1,pos:size(1) do 
      im[1][pos[i][2]][pos[i][1]] = 1 
   end   
   for i = 1,pos2:size(1) do 
      im[2][pos2[i][2]][pos2[i][1]] = 1 
   end   
   image.display{image=im, zoom=1}
   return pos, feat
end
