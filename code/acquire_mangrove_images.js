// load Landsat 7 Image Collection (surface reflectance)
var l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR');

// define dictionary for name and image center info

// Florida Images
// var dict = {"Name": "Florida_1", "centerPoint": [-80.90, 25.14]};
// var dict = {"Name": "Florida_2", "centerPoint": [-81.54, 24.66]};
// var dict = {"Name": "Florida_3", "centerPoint": [-80.36, 25.26]};
// var dict = {"Name": "Florida_4", "centerPoint": [-81.03, 25.40]};
// var dict = {"Name": "Florida_5", "centerPoint": [-81.13, 25.58]};

// Brazil
var dict = {"Name": "Brazil_1", "centerPoint": [-43.63, -2.50]};
// var dict = {"Name": "Brazil_2", "centerPoint": [-50.06, 1.67]};
// var dict = {"Name": "Brazil_3", "centerPoint": [-50.17, 0.83]};
// var dict = {"Name": "Brazil_4", "centerPoint": [-45.87, -1.15]};
// var dict = {"Name": "Brazil_5", "centerPoint": [-48.34, -25.29]};

// Cameroon
// var dict = {"Name": "Cameroon_1", "centerPoint": [9.57, 3.93]};

// set view center and zoom (12)
Map.setCenter(dict.centerPoint[0], dict.centerPoint[1], 12);

// define rectangle within which to work with satellite images
var half_width = 0.25
var half_height = 0.06
var rectangle = ee.Geometry.Rectangle([dict.centerPoint[0]-half_width,
                                       dict.centerPoint[1]-half_height,
                                       dict.centerPoint[0]+half_width,
                                       dict.centerPoint[1]+half_height,])


// grab satellite images which intersect defined rectangle
var spatialFiltered = l7.filterBounds(rectangle);
// print('spatialFiltered', spatialFiltered);

// grab images from the year 2000 and 2020
var images_2000 = spatialFiltered.filterDate('2000-01-01', '2000-12-31');
var images_2020 = spatialFiltered.filterDate('2020-01-01', '2020-12-31');
// print('images_2000', images_2000);
// print('images_2020', images_2020);

// define function for masking clouds out of images (from SR images)
var maskClouds = function(image){
  var pixel_qa = image.select('pixel_qa');
  // return image.updateMask(pixel_qa.eq(66)); // 66 or 130 - https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_C1-SurfaceReflectance-LEDAPS_ProductGuide-v3.pdf
  return image.updateMask(pixel_qa.eq(66).or(pixel_qa.eq(68))); // for water, it's 68 or 132 
};

// define collections of masked images
var images_2000_masked = images_2000.map(maskClouds);
var images_2020_masked = images_2020.map(maskClouds);

// defin function for adding NDVI band to images
var getNDVI = function(img){
  return img.addBands(img.normalizedDifference(['B4','B3']).rename('NDVI'));
};

var images_2000_ndvi = images_2000_masked.map(getNDVI);
var images_2020_ndvi = images_2020_masked.map(getNDVI);

// create composite image selecting on the maximum NDVI pixel throughout the set of images
var ndvi_2000_composite = images_2000_ndvi.qualityMosaic('NDVI').clip(rectangle);
var ndvi_2020_composite = images_2020_ndvi.qualityMosaic('NDVI').clip(rectangle);
// print('ndvi_2000_composite', ndvi_2000_composite);
// print('ndvi_2020_composite', ndvi_2000_composite);

// Visualize NDVI 
var ndviPalette = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
              '74A901', '66A000', '529400', '3E8601', '207401', '056201',
              '004C00', '023B01', '012E01', '011D01', '011301'];
Map.addLayer(ndvi_2000_composite.select('NDVI'), {min:0, max: 1, palette: ndviPalette}, 'ndvi_2000', false);
Map.addLayer(ndvi_2020_composite.select('NDVI'), {min:0, max: 1, palette: ndviPalette}, 'ndvi_2020', false);

// Visualize RGB images
var visParams = {bands: ['B3','B2','B1'], min: 150, max: 2000}
Map.addLayer(ndvi_2000_composite, visParams, 'rgb_2000', false);
Map.addLayer(ndvi_2020_composite, visParams, 'rgb_2020', false);

// load in Mangroves collection (consists of 1 image)
var mangroves_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS');
var mangroves = mangroves_coll.first().select(['1'], ['Mangroves']).clip(rectangle); // grab the image and rename the band

// Visualize the mangroves mapped out in 2000
var mangrovesVis = {min: 0, max: 1.0, palette: ['d40115']};
Map.addLayer(mangroves, mangrovesVis, 'Mangroves in Rectangle', false);
Map.addLayer(mangroves_coll, mangrovesVis, 'All Mangroves', false); // For visualizing beyond the bounds of the rectangle

// Add the mangroves band to the composite images
var composite_2000_with_mangroves = ndvi_2000_composite.addBands(mangroves, ['Mangroves']);
var composite_2020_with_mangroves = ndvi_2020_composite.addBands(mangroves, ['Mangroves']);
print('Full 2000 composite', composite_2000_with_mangroves);
print('Full 2020 composite', composite_2020_with_mangroves);

// Export 2000 image, with all bands, casted to float (increase size of image, but otherwise an error is thrown)
Export.image.toDrive({
  image: composite_2000_with_mangroves.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Mangroves', 'NDVI').toFloat(),
  description: dict.Name.concat('_2000'),
  scale: 30,
  region: composite_2000_with_mangroves.geometry().bounds(), // .geometry().bounds() needed for multipolygon
  folder: 'MangroveClassification',
  maxPixels: 2e9
});

// Export 2020 image, with all bands, casted to float (increase size of image, but otherwise an error is thrown)
Export.image.toDrive({
  image: composite_2020_with_mangroves.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Mangroves', 'NDVI').toFloat(),
  description: dict.Name.concat('_2020'),
  scale: 30, // meters per pixel
  region: composite_2020_with_mangroves.geometry().bounds(), // .geometry().bounds() needed for multipolygon
  folder: 'MangroveClassification',
  maxPixels: 2e9 // some large number, shouldn't be anywhere close to this
});





