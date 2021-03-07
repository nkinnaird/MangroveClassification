var l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR');

var spatialFiltered = l7.filterBounds(rectangle);
print('spatialFiltered', spatialFiltered);

var images_2000 = spatialFiltered.filterDate('2000-01-01', '2000-12-31');
print('images_2000', images_2000);

var images_2020 = spatialFiltered.filterDate('2020-01-01', '2020-12-31');
print('images_2020', images_2020);


// var visParams = {bands: ['B3', 'B2', 'B1'], max: 0.3};


// work with NDVI and normal mosaicing

var maskClouds = function(image){
  var pixel_qa = image.select('pixel_qa');
  // return image.updateMask(pixel_qa.eq(66)); // think this should be 66 or 130 - https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1370_L4-7_C1-SurfaceReflectance-LEDAPS_ProductGuide-v3.pdf
  return image.updateMask(pixel_qa.eq(66).or(pixel_qa.eq(68))); // for water, it's 68 or 132 
};

var images_2000_masked = images_2000.map(maskClouds);

// visualize the first image in the collection, pre- and post- mask
var visParams = {bands: ['B3','B2','B1'], min: 150, max: 2000}

// Map.addLayer(ee.Image(images_2000_masked.first()), visParams, 'clouds masked', false)
// Map.addLayer(ee.Image(images_2000.first()), visParams, 'original', false)


var getNDVI = function(img){
  return img.addBands(img.normalizedDifference(['B4','B3']).rename('NDVI'));
};

var images_2000_ndvi = images_2000_masked.map(getNDVI);






var ndvi_2000_composite = images_2000_ndvi.qualityMosaic('NDVI');//.clip(rectangle);
print('ndvi_2000_composite', ndvi_2000_composite);

// Visualize NDVI
var ndviPalette = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
               '74A901', '66A000', '529400', '3E8601', '207401', '056201',
               '004C00', '023B01', '012E01', '011D01', '011301'];
Map.addLayer(ndvi_2000_composite.select('NDVI'), {min:0, max: 1, palette: ndviPalette}, 'ndvi_2000');

Map.addLayer(ndvi_2000_composite, visParams, 'true color composite', false);



var mangroves_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS');

// var mangroves = mangroves_coll.first();//.clip(rectangle);
var mangroves = mangroves_coll.first().select(['1'], ['Mangroves']);//.clip(rectangle);
// mangroves.rename(['Mangroves'])
print('mangroves image', mangroves);

var mangrovesVis = {
  min: 0,
  max: 1.0,
  palette: ['d40115'],
};
// Map.setCenter(-80.9014, 25.1357, 12);
Map.addLayer(mangroves, mangrovesVis, 'Mangroves', false);


// var addMangroveBand = function(img){
//   return img.addBands(mangroves(['1']).rename('Mangroves'));
// };

var ndvi_2000_composite_with_mangroves = ndvi_2000_composite.addBands(mangroves, ['Mangroves']);
print('mangroves image added', ndvi_2000_composite_with_mangroves);



Export.image.toDrive({
  // image: ndvi_2000_composite.select('NDVI'),
  // image: ndvi_2000_composite.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'),//, 'NDVI'),
  // image: ndvi_2000_composite_with_mangroves.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Mangroves'),//, 'NDVI'),
  image: ndvi_2000_composite_with_mangroves.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Mangroves', 'NDVI').toFloat(),
  description: 'ndvi_test',
  scale: 30,
  // scale: 1000,
  // region: watershed.geometry().bounds(), // .geometry().bounds() needed for multipolygon
  // region: ndvi_2000_composite.geometry().bounds(), // .geometry().bounds() needed for multipolygon
  // crs: 'EPSG:5070',
  folder: 'MangroveClassification',
  maxPixels: 2e9
});






print('Current center: ', Map.getCenter());
print('Current zoom: ', Map.getZoom());