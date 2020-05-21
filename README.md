# Adversarial Attack

This repo includes implementation of [FGS](https://arxiv.org/abs/1412.6572) algorithm, targeted and untargeted version.

## Fast Gradient Sign Untargeted

For untargeted FGS, we only want to fool the model not to predict image(A) as A.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Eel</strong> (390) <br/> Confidence: 0.96 </td>
			<td width="27%" align="center"> Adversarial Noise </td>
			<td width="27%" align="center"> Predicted as <strong>Blowfish</strong> (397) <br/> Confidence: 0.81 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/images/eel.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/untargeted_adv_noise_from_390_to_397.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/untargeted_adv_img_from_390_to_397.jpg"> </td>
		</tr>
	</tbody>
</table>

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Snowbird</strong> (13) <br/> Confidence: 0.99 </td>
			<td width="27%" align="center"> Adversarial Noise </td>
			<td width="27%" align="center"> Predicted as <strong>Chickadee</strong> (19) <br/> Confidence: 0.95 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/images/bird.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/untargeted_adv_noise_from_13_to_19.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/untargeted_adv_img_from_13_to_19.jpg"> </td>
		</tr>
	</tbody>
</table>

## Fast Gradient Sign Targeted

Given a targeted class T, when the model predict image(A) as T, iteration finished.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Apple</strong> (948) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Adversarial Noise </td>
			<td width="27%" align="center"> Predicted as <strong>Rock python</strong> (62) <br/> Confidence: 0.16 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/images/apple.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/targeted_adv_noise_from_948_to_62.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/targeted_adv_img_from_948_to_62.jpg"> </td>
		</tr>
	</tbody>
</table>

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Apple</strong> (948) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Adversarial Noise </td>
			<td width="27%" align="center"> Predicted as <strong>Mud turtle</strong> (35) <br/> Confidence: 0.54 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/images/apple.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/targeted_adv_noise_from_948_to_35.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/cowarder/adversarial-attack/master/generated/targeted_adv_img_from_948_to_35.jpg"> </td>
		</tr>
	</tbody>
</table>

