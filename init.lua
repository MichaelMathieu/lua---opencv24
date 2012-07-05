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

function opencv24.CreateFREAK(orientedNormalization, scaleNormalization,
			      patternSize, nOctave, trainedPairs)
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

function opencv24.FREAK_testme()
   local iFREAK = opencv24.CreateFREAK(true, true, 22, 4)
   local im = image.lena()
   local im2 = image.rotate(im, 0.1)
   local trainedPairs = opencv24.TrainFREAK({im}, iFREAK, 40, 0.7)
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
      local x1 = freaks.pos[matches[i][1]+1][1]
      local y1 = freaks.pos[matches[i][1]+1][2]
      local x2 = freaks2.pos[matches[i][2]+1][1]
      local y2 = freaks2.pos[matches[i][2]+1][2]
      draw.line(disp, x1, y1, x2+imb:size(3), y2, 0, 0, 1)
   end
   image.display{image=disp, zoom=1}
   opencv24.DeleteFREAK(iFREAK)
end

function opencv24.testme()
   print("OpenCV 2.4: testme...")
end
