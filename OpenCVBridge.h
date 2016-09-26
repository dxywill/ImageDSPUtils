//
//  OpenCVBridge.h
//  LookinLive
//
//  Created by Eric Larson on 8/27/15.
//  Copyright (c) 2015 Eric Larson. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>

@interface OpenCVBridge : NSObject

-(void) setImage:(CIImage*)ciFrameImage
      withBounds:(CGRect)rect
      andContext:(CIContext*)context;

-(CIImage*)getImage;

-(void)processImage;


@end
