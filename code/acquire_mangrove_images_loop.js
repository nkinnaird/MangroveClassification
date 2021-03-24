// This file should be run on Google Earth Engine. (Perhaps this can be passed to the python API but I just used the web app API.)
// This file will load Landsat 7 satellite images, select a center for the view, composite images based on location and time, and lastly export the image to Google Drive.
// This file is slightly different from the non-loop version, in that it loops through years and several images for each year.

// load Landsat 7 Image Collection (surface reflectance)
var l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR');

// define dictionary for name and image center info

// Florida Images
var dict = {"Name": "Florida_1", "centerPoint": [-80.90, 25.14]};
// var dict = {"Name": "Florida_2", "centerPoint": [-81.54, 24.66]};
// var dict = {"Name": "Florida_3", "centerPoint": [-80.36, 25.26]};
// var dict = {"Name": "Florida_4", "centerPoint": [-81.03, 25.40]};
// var dict = {"Name": "Florida_5", "centerPoint": [-81.13, 25.58]};
// var dict = {"Name": "Florida_6", "centerPoint": [-81.26, 25.74]};
// var dict = {"Name": "Florida_7", "centerPoint": [-81.50, 25.90]};

// Cuba
// var dict = {"Name": "Cuba_1", "centerPoint": [-81.83, 22.34]};
// var dict = {"Name": "Cuba_2", "centerPoint": [-81.03, 23.07]};
// var dict = {"Name": "Cuba_3", "centerPoint": [-80.03, 22.95]};
// var dict = {"Name": "Cuba_4", "centerPoint": [-78.56, 22.21]};


// Turks and Caicos
// var dict = {"Name": "TurksAndCaicos_1", "centerPoint": [-71.84, 21.76]};

// Brazil
// var dict = {"Name": "Brazil_1", "centerPoint": [-43.63, -2.50]};
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


// define function for masking clouds out of images (from SR images)
var maskClouds = function(image){
  var pixel_qa = image.select('pixel_qa');
  // return image.updateMask(pixel_qa.eq(66)); // 66 or 130 - https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_C1-SurfaceReflectance-LEDAPS_ProductGuide-v3.pdf
  return image.updateMask(pixel_qa.eq(66).or(pixel_qa.eq(68))); // for water, it's 68 or 132 
};

// define function for adding NDVI band to images
var getNDVI = function(img){
  return img.addBands(img.normalizedDifference(['B4','B3']).rename('NDVI'));
};

// NDVI vis
var ndviPalette = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
              '74A901', '66A000', '529400', '3E8601', '207401', '056201',
              '004C00', '023B01', '012E01', '011D01', '011301'];


// load in Mangroves collection (consists of 1 image)
var mangroves_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS');
var mangroves = mangroves_coll.first().select(['1'], ['Mangroves']).clip(rectangle); // grab the image and rename the band



// loop through years from 2001 to 2019 (already got 2020)
for(var i = 1; i <= 19; i++){
  var year = 2000+i;
  var low_date = year + '-01-01';
  var high_date = year + '-12-31';
  
  // grab images from specific year
  var filtered_images = spatialFiltered.filterDate(low_date, high_date);

  // define collections of masked images
  var filtered_images_masked = filtered_images.map(maskClouds);
  var filtered_images_ndvi = filtered_images_masked.map(getNDVI);

  // create composite image selecting on the maximum NDVI pixel throughout the set of images
  var ndvi_composite = filtered_images_ndvi.qualityMosaic('NDVI').clip(rectangle);

  // Visualize NDVI 
  Map.addLayer(ndvi_composite.select('NDVI'), {min:0, max: 1, palette: ndviPalette}, ('ndvi_'+year), false);

  // Add the mangroves band to the composite images
  var composite_with_mangroves = ndvi_composite.addBands(mangroves, ['Mangroves']);
  print('Year: ', year, " image: ", composite_with_mangroves)

  // Export image, with all bands, casted to float (increase size of image, but otherwise an error is thrown)
  Export.image.toDrive({
    image: composite_with_mangroves.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Mangroves', 'NDVI').toFloat(),
    description: dict.Name.concat(('_'+year)),
    scale: 30,
    region: composite_with_mangroves.geometry().bounds(), // .geometry().bounds() needed for multipolygon
    folder: 'MangroveClassification',
    maxPixels: 2e9
  });


}












// Visualize the mangroves mapped out in 2000
var mangrovesVis = {min: 0, max: 1.0, palette: ['d40115']};
Map.addLayer(mangroves, mangrovesVis, 'Mangroves in Rectangle', false);
Map.addLayer(mangroves_coll, mangrovesVis, 'All Mangroves', false); // For visualizing beyond the bounds of the rectangle
















