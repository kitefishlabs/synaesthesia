s.boot
~datafile = File.open("/Users/kfl/dev/python/synaesthesia/tripod/Vertigo.mov.motionbrightness", "rb");
~dataArray = (~datafile.length div:4).collect({ ~datafile.getInt32LE.asFloat / 1000.0 });
~datafile.close;

~dataArray.size / 640
~timeframes = (~dataArray.size / 640).asInt;
~dataArray = ~dataArray.reshape(~timeframes, 64, 10);

(
SynthDef(\klanker, {| freqs=#[100,200,300,400,500,600,700,800], amps=#[1,1,1,1,1,1,1,1], pos=0|
	var env = EnvGen.ar(Env.linen(0.01, 0.07, 0.01, curve:\sine), doneAction:2);
	Out.ar(0, Pan2.ar(DynKlank.ar(`[freqs, amps], WhiteNoise.ar(0.1) * env), pos));

}).load(s);

SynthDef(\klanger, {| freqs=#[100,200,300,400,500,600,700,800], amps=#[1,1,1,1,1,1,1,1], pos=0|
	var env = EnvGen.ar(Env.linen(0.01, 0.07, 0.01, curve:\sine), doneAction:2);
	Out.ar(0, Pan2.ar(DynKlang.ar(`[freqs, amps]) * env, pos));

}).load(s);


SynthDef(\klanger2, {| freq0=100,freq1=200,freq2=300,freq3=400,freq4=500,freq5=600,freq6=700,freq7=800, amp0=0,amp1=0,amp2=0,amp3=0,amp4=0,amp5=0,amp6=0,amp7=0, pos=0|
	// var env = EnvGen.ar(Env.linen(0.01, 70, 0.01, curve:\sine), doneAction:0);
	var chain = Mix(SinOsc.ar(Lag.kr([freq0,freq1,freq2,freq3,freq4,freq5,freq6,freq7], 0.2), 0, Lag.kr([amp0,amp1,amp2,amp3,amp4,amp5,amp6,amp7], 0.2)));
	Out.ar(0, Pan2.ar(chain, pos));

}).load(s);

)

// Env.linen(0.2, 0.4, 0.2, curve:\sine).plot

// Synth(\klanker);


~voices[13].setn(\freqs, Array.with(300, 400, 0, 0, 0, 0, 0, 0))
~voices[13].setn(\amps, Array.with(0.8, 0.7, 0, 0, 0, 0, 0, 0))

(
~voices = Array.series(64).collect({ |v| Synth(\klanger2, [\pos, (v / 32.0) - 1.0], addAction:\addToTail) });
s.queryAllNodes;

Routine {
	1.wait;
	~timeframes.do({ |frame|

		64.do( {|vox|
			var bright = (~dataArray[frame][vox][8] * 72).round(1);
			var fund = (bright + 24).midicps;
			// var freqs = [1,2,3,4,5,6,7,8] * fund;
			// var freqs = [1,3,5,7,9,11,13,15] * fund;
			var freqs = [1,2,3,4,6,8,12,16] * fund;
			var amps = ~dataArray[frame][vox][..7] * [1, 0.8,0.6, 0.4, 0.2, 0.25, 0.075, 0.05];

			// freqs.postln;
			amps.postln;

			~voices[vox].setn(\freq0, freqs.asArray, \amp0, amps.asArray);

		});
		// 0.0416.wait;
		0.0166.wait;
	})
}.play;


// ~voices.do( {|vox| vox.free });
)
s.queryAllNodes

(
Routine {
	1.wait;
	~timeframes.do({ |frame|

		64.do( {|vox|
			var bright = (~dataArray[frame][vox][8] * 36).round(0.33333333333);
			var fund = (bright + 24).midicps;
			// var freqs = [1,2,3,4,5,6,7,8] * fund;
			// var freqs = [1,3,5,7,9,11,13,15] * fund;
			var freqs = [1,2,3,4,6,8,12,16] * fund;
			var amps = ~dataArray[frame][vox][..7] * [1, 0.8,0.6, 0.4, 0.2, 0.25, 0.075, 0.05];

			(amps.sum > 0.1).if {
				Synth(\klanger, ['freqs', freqs, 'amps', amps, 'pos', (vox / 32.0) - 1.0]);
			};
		});
		0.0416.wait;
	})
}.play;
)

(
Routine {
	1.wait;
	~timeframes.do({ |frame|

		64.do( {|vox|
			var bright = (~dataArray[frame][vox][8] * 36).round(0.33333333333);
			var fund = (bright + 24).midicps;
			// var freqs = [1,2,3,4,5,6,7,8] * fund;
			// var freqs = [1,3,5,7,9,11,13,15] * fund;
			var freqs = [1,2,3,4,6,8,12,16] * fund;
			var amps = ~dataArray[frame][vox][..7] * [1, 0.8,0.6, 0.4, 0.2, 0.25, 0.075, 0.05];

			(amps.sum > 0.1).if {
				Synth(\klanker, ['freqs', freqs, 'amps', amps, 'pos', (vox / 32.0) - 1.0]);
			};
		});
		0.0416.wait;
	})
}.play;
)



