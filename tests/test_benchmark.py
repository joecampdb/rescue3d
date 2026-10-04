import unittest
import numpy as np
import jax
import jax.numpy as jnp
from world import World,DIRECTIONS,rotation,at,BUDGET
from experiment import model_input,episode
from models import init,online,initial_state,metadata,trainer


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params={k:init(11,k) for k in ['ff','gru','reservoir','spiking']}

    def test_world_has_reachable_targets_on_multiple_levels(self):
        w=World(3)
        self.assertTrue(all(at(w.reachable,p) for p in w.targets))
        self.assertGreater(np.argwhere(w.reachable)[:,0].max(),4)

    def test_identical_seed_same_scene_and_sensors(self):
        a=World(4);b=World(4)
        np.testing.assert_array_equal(a.blocked,b.blocked)
        x=a.observe(True);y=b.observe(True)
        np.testing.assert_array_equal(x['points'],y['points'])
        np.testing.assert_array_equal(x['cue'],y['cue'])

    def test_point_cloud_is_three_dimensional_and_occluded(self):
        w=World(4);obs=w.observe()
        self.assertEqual(obs['points'].shape,(64,3))
        self.assertEqual(np.linalg.matrix_rank(obs['points'][obs['mask']>0]),3)
        # Negative-x cardinal beam first meets the neighboring boundary voxel.
        self.assertLess(obs['ranges'][1],1.)

    def test_collision_and_budget(self):
        w=World(3);w.act(1)
        self.assertEqual(tuple(w.pose),w.base);self.assertEqual(w.collisions,1)
        self.assertEqual(w.energy,BUDGET-1)

    def test_no_privileged_labels_in_model_inputs(self):
        w=World(3);obs=w.observe();a=model_input(obs)
        obs['labels'][:]=999;obs['estimate'][:]=99;b=model_input(obs)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(set(a),{'points','mask','cue','delta','query','reset'})

    def test_invariant_networks_under_joint_rotation(self):
        w=World(6);obs=w.observe(True);obs['delta']=np.array([.3,.5,.2],np.float32)
        a=model_input(obs);b=model_input(obs,rotation(45))
        for kind,(p,_) in self.params.items():
            # Carry a real history: rotate every observation, preserve channel identities.
            sa=initial_state();sb=initial_state()
            for _ in range(3):
                sa,ya,_=online(p,sa,a,'invariant',kind);sb,yb,_=online(p,sb,b,'invariant',kind)
            np.testing.assert_allclose(ya,yb,atol=1e-4,rtol=1e-4)

    def test_point_permutation(self):
        w=World(5);obs=model_input(w.observe());perm=np.random.default_rng(4).permutation(64)
        other={**obs,'points':obs['points'][perm],'mask':obs['mask'][perm]}
        p,_=self.params['gru']
        for enc in ['xyz','invariant']:
            _,a,_=online(p,initial_state(),obs,enc,'gru');_,b,_=online(p,initial_state(),other,enc,'gru')
            np.testing.assert_allclose(a,b,atol=1e-5)

    def test_all_missing_cloud_is_finite(self):
        obs=model_input(World(5).observe());obs['mask'][:]=0
        for kind,(p,_) in self.params.items():
            _,y,_=online(p,initial_state(),obs,'invariant',kind)
            self.assertTrue(np.isfinite(y).all())

    def test_caps_and_weight_matching(self):
        counts=[]
        for k,(p,w) in self.params.items():
            m=metadata(p,w,k);counts.append(m['parameters'])
            self.assertLessEqual(m['parameters'],4096);self.assertLessEqual(m['persistent_state_floats'],96)
            self.assertLessEqual(m['dense_macs'],2_000_000)
        self.assertLess((max(counts)-min(counts))/min(counts),.005)

    def test_feedforward_does_not_use_history(self):
        p,_=self.params['ff'];obs=model_input(World(5).observe())
        _,a,_=online(p,initial_state(),obs,'xyz','ff')
        _,b,_=online(p,jnp.ones((6,16)),obs,'xyz','ff')
        np.testing.assert_array_equal(a,b)

    def test_spikes_are_binary_and_voltage_persists(self):
        p,_=self.params['spiking'];obs=model_input(World(5).observe())
        state,_,_=online(p,initial_state(),obs,'invariant','spiking')
        self.assertTrue(np.isin(np.asarray(state[:,8:]),[0,1]).all())
        self.assertTrue(np.any(np.asarray(state[:,:8])!=0))

    def test_state_reset_discards_history(self):
        obs=model_input(World(5).observe());obs['reset']=np.float32(1)
        for kind,(p,_) in self.params.items():
            _,a,_=online(p,initial_state(),obs,'invariant',kind)
            _,b,_=online(p,jnp.ones((6,16)),obs,'invariant',kind)
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
