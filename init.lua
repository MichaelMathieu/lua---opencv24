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
   local im_cv = torch.ByteTensor(im:size(2), im:size(3), 3)
   im.libopencv24.TH2CVImage(im, im_cv)
   return im_cv
end

function opencv24.CV2THImage(im_cv)
   local im = torch.Tensor(3, im_cv:size(1), im_cv:size(2))
   im.libopencv24.CV2THImage(im_cv, im)
   return im
end

--------------------------------------------------------------------------------
-- Tracking
--

function opencv24.TrackPointsLK(...)
   local self = {}
   xlua.unpack_class(
      self, {...}, 'sfm2.getEgoMotion', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
      {arg='maxPoints', type='number', help='Maximum number of tracked points', default=500},
      {arg='pointsQuality',type='number',help='Minimum quality of trackedpoints',default=0.02},
      {arg='pointsMinDistance', type='number',
       help='Minumum distance between two tracked points', default=3.0},
      {arg='featuresBlockSize', type='number',
       help='opencv GoodFeaturesToTrack block size', default=20},
      {arg='trackerWinSize', type='number',
       help='opencv calcOpticalFlowPyrLK block size', default=10},
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
      self, {...}, 'sfm2.getEgoMotion', help_desc,
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
-- FREAK
--

function opencv24.CreateFREAK(orientedNormalization, scaleNormalization,
			      patternSize, nOctave, trainedPairs)
   orientedNormalization = orientedNormalization or true
   scaleNormalization = scaleNormalization or true
   patternSize = patternSize or 22
   nOctave = nOctave or 4
   trainedPairs = trainedPairs or torch.IntTensor()
   return libopencv24.CreateFREAK(orientedNormalization, scaleNormalization,
				  patternSize, nOctave, trainedPairs)
end

function opencv24.DeleteFREAK(iFREAK)
   libopencv24.DeleteFREAK(iFREAK)
end

function opencv24.ComputeFREAK(im, detection_threshold, iFREAK)
   local freaks = {}
   freaks.descs = torch.ByteTensor()
   freaks.pos = torch.FloatTensor()
   im.libopencv24.ComputeFREAK(im, freaks.descs, freaks.pos, detection_threshold, iFREAK);
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
   local pairs = torch.IntTensor()
   images[1].libopencv24.TrainFREAK(images, pairs, iFREAK, keypoints_threshold,
				    correlation_threshold)
   return pairs
end

--------------------------------------------------------------------------------
-- Test/Example
--

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
   local size = 22*6
   local iFREAK = opencv24.CreateFREAK(true, true, size, 4)
   --local im = image.lena()
   --local im2 = image.rotate(im, 0.1)
   local im  = image.load('/home/myrhev/NYU/depth-estimation/radial/data/no-risk/part1/images/000000001.jpg')
   local im2 = image.load('/home/myrhev/NYU/depth-estimation/radial/data/no-risk/part1/images/000000002.jpg')
   --local trainedPairs = opencv24.TrainFREAK({im}, iFREAK, 40, 0.7)
   --local iFREAK = opencv24.CreateFREAK(true, true, size, 4, trainedPairs)
   --local iFREAK = opencv24.CreateFREAK(true, true, 22, 4)
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
